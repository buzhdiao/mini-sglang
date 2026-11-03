from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias  # 类型注解工具：NamedTuple（具名元组）、NoReturn（永不返回）、TypeAlias（类型别名）

import torch  # PyTorch：CUDA 流管理、张量构造与 GPU 高级索引
from minisgl.core import Batch, Req  # 核心结构：批次与请求
from minisgl.env import ENV  # 环境变量开关（如禁用重叠调度）
from minisgl.message import (  # 消息类型：批量/退出/用户/中止消息与去 token 化结果
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
from minisgl.utils import init_logger, load_tokenizer  # 日志工具与分词器加载

from .cache import CacheManager  # KV 缓存管理器：页级分配与前缀缓存
from .config import SchedulerConfig  # 调度器配置
from .decode import DecodeManager  # 解码阶段管理器
from .io import SchedulerIOMixin  # I/O 混入类：ZMQ 收发与多 rank 同步
from .prefill import ChunkedReq, PrefillManager  # 分块请求与预填充管理器
from .table import TableManager  # 请求槽位管理器

if TYPE_CHECKING:  # 仅静态类型检查时导入，避免运行时循环依赖
    from minisgl.engine import BatchSamplingArgs, ForwardOutput  # 引擎的批量采样参数与前向输出类型


logger = init_logger(__name__)  # 模块级日志器

Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]  # 二维索引别名：(行索引, 列索引) 张量对，用于张量高级索引


# For overlap scheduling, we also need to cache some other data to avoid IMA
# 重叠调度下需把一次前向的全部输入缓存为快照，延迟到下一轮才消费其结果，避免非法内存访问（IMA）
class ForwardInput(NamedTuple):
    """一次前向所需的完整输入快照（NamedTuple：不可变、可按序解包）"""

    batch: Batch  # 本批次的批次对象
    sample_args: BatchSamplingArgs  # 批量采样参数（由 sampler.prepare 生成）
    input_tuple: Indice2D  # (token_mapping, positions)  # 读索引：从 token 池按 (槽位, 位置) 取输入 token
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)  # 写索引：新 token 写回 token 池的 (槽位, 位置)；-1 表示该请求不写


ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"  # 一次前向的 (输入快照, 输出) 对：跨轮次传递的在途数据


class Scheduler(SchedulerIOMixin):
    """调度器主类：接收请求、调度 prefill/decode 批次、驱动引擎前向并处理输出（继承 I/O 混入）"""

    def __init__(self, config: SchedulerConfig):
        from minisgl.engine import Engine  # 函数内延迟导入：打破模块间循环依赖

        self.engine = Engine(config)  # 创建推理引擎（模型、KV 池、页表、CUDA graph、采样器等）

        # use another stream to overlap metadata processing with computation  # 用独立 CUDA 流让 CPU 元数据处理与 GPU 计算重叠
        self.device = self.engine.device  # GPU 设备（rank 即本地卡号）
        self.stream = torch.cuda.Stream(device=self.device)  # 调度器专属 CUDA 流：跑页表/索引等元数据准备
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)  # 进入引擎流的上下文管理器（前向计算在此流）
        torch.cuda.set_stream(self.stream)  # 把当前线程默认流切到调度器流（之后的张量操作默认在此流）

        # initialize other managers  # 初始化各子管理器
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)  # 槽位管理（共享引擎页表）
        self.cache_manager = CacheManager(  # KV 缓存管理（共享页表；页级分配 + 前缀缓存）
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        self.decode_manager = DecodeManager(config.page_size)  # 解码管理（页大小用于在途 token 预留估算）
        self.prefill_manager = PrefillManager(  # 预填充管理（组合缓存/槽位/解码三个管理器）
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access  # 常用别名与调度器自身状态
        self.finished_reqs: Set[Req] = set()  # 上一轮已完成的请求集合：防重叠调度下同一请求被二次释放
        self.tokenizer = load_tokenizer(config.model_path)  # 加载分词器（本进程需要 eos id 判定完成）
        self.eos_token_id = self.tokenizer.eos_token_id  # EOS 结束符 token id
        self.token_pool = self.table_manager.token_pool  # token 池别名（GPU 上存各请求 token id 的张量）
        self.prefill_budget = config.max_extend_tokens  # prefill 的 token 预算
        # self.config = config  # （原有注释）曾保存 config 引用，现已停用

        # Initialize the I/O mixin  # 初始化 I/O 混入（ZMQ 通道；放最后：依赖已就绪的 tp_cpu_group）
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        # 空闲钩子：调度器无事可做、将要阻塞等消息前调用，执行后台维护任务
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")  # 提示进入空闲等待
        self.cache_manager.check_integrity()  # 校验缓存页守恒（空闲页+缓存页=总页数），尽早发现泄漏

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # 重叠调度主循环：GPU 执行本轮批次期间，CPU 同时处理上一轮结果并调度下一轮
        blocking = not (  # 是否阻塞等消息：仅当既无在途批次、又无任何可调度任务时才阻塞
            last_data is not None  # don't block if we have a batch to be processed  # 有在途批次则不阻塞
            or self.prefill_manager.runnable  # 或 prefill 队列非空
            or self.decode_manager.runnable  # 或有解码中的请求
        )
        for msg in self.receive_msg(blocking=blocking):  # 接收并逐条处理消息（空闲时阻塞等待新请求）
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()  # 调度并准备下一批的输入元数据（跑在调度器流，异步）
        ongoing_data = None  # 本轮发起的在途前向数据（下一轮再收结果）
        if forward_input is not None:  # 有批次需要执行
            with self.engine_stream_ctx:  # run the batch in the engine's stream  # 切到引擎流发起前向
                self.engine.stream.wait_stream(self.stream)  # 引擎流等待调度器流的元数据准备完成
                ongoing_data = (forward_input, self._forward(forward_input))  # 发起前向（异步发起，不等待完成）

        self._process_last_data(last_data)  # 处理上一轮前向的输出（此刻 GPU 大概率已完成）
        return ongoing_data  # 在途数据传给下一轮循环，形成流水线

    def normal_loop(self) -> None:
        """非重叠调度主循环：同步地 收消息 -> 调度 -> 前向 -> 处理结果"""
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)  # 无可调度任务才阻塞等消息
        for msg in self.receive_msg(blocking=blocking):  # 接收消息
            self._process_one_msg(msg)

        forward_input = self._schedule_next_batch()  # 调度下一批
        ongoing_data = None
        if forward_input is not None:  # 有批次则同步执行并保存结果
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)  # 立即处理本轮结果（无流水线重叠）

    @torch.inference_mode()  # 推理模式装饰器：禁用自动求梯度追踪（比 no_grad 更彻底），省显存提速
    def run_forever(self) -> NoReturn:
        """调度器主入口：按环境变量选择重叠/非重叠模式，永久循环直到收到退出消息"""
        if ENV.DISABLE_OVERLAP_SCHEDULING:  # 环境变量禁用重叠调度
            with self.engine_stream_ctx:  # 非重叠模式：全程在引擎流中执行
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()  # 同步循环
        else:
            assert torch.cuda.current_stream() == self.stream  # 重叠模式要求当前流为调度器流（__init__ 已设置）
            data = None  # 初始无在途数据
            while True:
                data = self.overlap_loop(data)  # 重叠循环：在途数据在轮次间串联

    def shutdown(self) -> None:
        """关闭调度器：同步 GPU -> 同步各 rank -> 关闭引擎"""
        torch.cuda.synchronize(self.device)  # 等待该设备所有流上的计算与拷贝完成
        self.sync_all_ranks()  # CPU 侧全 rank 屏障：确保各 TP rank 一致地走到这里
        self.engine.shutdown()  # 关闭引擎并释放资源

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        # 处理上一轮前向的输出：追加 token、判定完成、缓存前缀/释放资源、回复 tokenizer
        if last_data is None:  # 无在途数据（启动初期或本轮未跑批次）
            return

        # 解包在途数据：批次对象 + 前向输出（next_tokens 的 GPU->CPU 拷贝句柄与完成事件）
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()  # 等待异步的 GPU->CPU token 拷贝完成（CPU 读取前的同步点）
        reply: List[DetokenizeMsg] = []  # 回复给 tokenizer 的消息列表
        new_finished_reqs: Set[Req] = set()  # 本轮新完成的请求集合
        with self.cache_manager.lazy_free_region():  # 延迟释放区域：本块内的页释放先收集，退出时统一归还（避免与在途前向竞态）
            for i, req in enumerate(batch.reqs):  # 逐请求处理新输出
                if isinstance(req, ChunkedReq):  # 分块请求 prefill 未完成，本轮无新 token 可处理
                    continue
                next_token = next_tokens_cpu[i]  # 该请求新采样出的 token（CPU 标量张量）
                req.append_host(next_token.unsqueeze(0))  # 追加到 CPU 侧序列（unsqueeze 保持 1 维形状约定）
                next_token = int(next_token.item())  # 转为 Python int 便于比较与消息传递
                finished = not req.can_decode  # 达到最大生成长度 => 完成
                if not req.sampling_params.ignore_eos:  # 未配置"忽略 EOS"时
                    finished |= next_token == self.eos_token_id  # 采样到 EOS 也判定完成
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))  # 构造去 token 化消息

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                # NOTE：重叠调度可能使同一请求的结果被处理两次，用完成集合去重，防止二次释放
                if finished and req not in self.finished_reqs:  # 本轮新完成（上一轮未记录过）的请求
                    self.decode_manager.remove_req(req)  # 从解码运行集合移除
                    self._free_req_resources(req)  # 释放槽位与 KV 资源
                    new_finished_reqs.add(req)  # 记入本轮完成集合
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix  # prefill 完成（未结束）的请求：KV 前缀入缓存
                    self.cache_manager.cache_req(req, finished=False)  # 插入前缀缓存供后续请求复用（保留尾段）

        self.finished_reqs = new_finished_reqs  # 整体替换完成集合（只保留最近一轮，窗口覆盖重叠深度即可去重）
        self.send_result(reply)  # 把回复消息发送给 tokenizer 进程

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        # 消息分发：按类型处理批量/退出/用户/中止四类消息
        if isinstance(msg, BatchBackendMsg):  # 批量消息：拆包后递归逐条处理
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):  # 退出消息：抛键盘中断终止 run_forever 主循环
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):  # 用户请求消息：合法性检查后入 prefill 队列
            logger.debug_rank0("Received user msg: %s", msg)
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len  # 输入长度与模型最大序列长度
            max_output_len = max_seq_len - input_len  # 剩余可生成的最大输出长度
            if max_output_len <= 0:  # 输入已超上限：直接丢弃该请求
                return logger.warning_rank0(  # return 日志函数返回的 None，等效 return None
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:  # 期望输出超出剩余空间：截断到上限
                msg.sampling_params.max_tokens = max_output_len  # 就地修改采样参数（Msg 对象可变）
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)  # 加入 prefill 等待队列
        elif isinstance(msg, AbortBackendMsg):  # 中止消息：按 uid 取消请求
            logger.debug_rank0("Aborting request %d", msg.uid)
            req_to_free = self.prefill_manager.abort_req(msg.uid)  # 先尝试从 prefill 等待队列中止
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)  # 再尝试从解码集合中止（返回命中的 Req）
            if req_to_free is not None:  # 找到占用资源的请求
                self._free_req_resources(req_to_free)  # 释放其槽位与 KV 资源
        else:
            logger.error(f"Unknown message type: {type(msg)}")  # 未知消息类型：记录错误并终止进程
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        # 释放请求占用的全部资源：槽位归还 + KV 前缀入库/尾段释放
        self.table_manager.free(req.table_idx)  # 归还请求槽位
        self.cache_manager.cache_req(req, finished=True)  # 前缀插入缓存，并按 finished=True 释放尾段页

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # 批次执行前的准备：补齐 -> 分配 KV 页 -> 构造读写索引 -> 准备注意力元数据（均在调度器流上异步执行）
        self.engine.graph_runner.pad_batch(batch)  # 按 CUDA graph 支持的形状补齐批次（可能追加 dummy req）
        self.cache_manager.allocate_paged(batch.reqs)  # 为各请求新增 token 分配 KV 页并写页表
        batch.positions = _make_positions(batch, self.device)  # 构造每个 token 的位置索引张量
        input_mapping = _make_input_tuple(batch, self.device)  # 构造读索引：(槽位, 位置) 用于从 token 池取输入
        write_mapping = _make_write_tuple(batch, self.device)  # 构造写索引：(槽位, 位置/-1) 用于写回新 token
        batch.out_loc = self.engine.page_table[input_mapping]  # 查页表：得到新 KV 写入缓存池的物理位置
        self.engine.attn_backend.prepare_metadata(batch)  # 注意力后端准备核函数元数据（如页表指针、序列长度）
        return ForwardInput(  # 打包为不可变快照（重叠调度下要跨轮次保存）
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),  # 由采样器生成批量采样参数
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # 调度策略：prefill 优先（先消化等待队列），无 prefill 批次则做解码
        # TODO: support other policies: e.g. DECODE first  # TODO：支持其他策略（如解码优先/混合调度）
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)  # 先尝试调度 prefill 批次
            or self.decode_manager.schedule_next_batch()  # prefill 无批次则调度 decode 批次
        )
        return self._prepare_batch(batch) if batch else None  # 有批次则准备元数据，否则返回 None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        # 执行一次前向：取输入 -> 引擎计算+采样 -> 新 token 写回 token 池 -> 维护解码集合
        batch, sample_args, input_mapping, output_mapping = forward_input  # 解包输入快照
        batch.input_ids = self.token_pool[input_mapping]  # 按 (槽位, 位置) 索引从 token 池取出输入 token（GPU 高级索引）
        forward_output = self.engine.forward_batch(batch, sample_args)  # 引擎执行前向与采样，产出 next_tokens
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu  # 新 token 按写索引写回 token 池（下一步 decode 的输入）
        self.decode_manager.filter_reqs(forward_input.batch.reqs)  # 本批请求并入解码集合（自动过滤不可解码者，如分块请求）
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    # 构造位置张量：每个请求新增 token 的绝对位置区间 [cached_len, device_len)，CPU 填充后异步上 GPU
    needed_size = sum(r.extend_len for r in batch.padded_reqs)  # 总 token 数（含补齐请求）
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)  # 锁页内存缓冲（支持 non_blocking 拷贝）
    offset = 0  # 当前填充偏移
    for req in batch.padded_reqs:  # 逐请求填充其位置区间
        length = req.extend_len  # 该请求新增 token 数
        torch.arange(  # 生成连续位置 [cached_len, device_len)
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],  # 直接写入缓冲区对应片段
        )
        offset += length  # 前进偏移
    return indices_host.to(device, non_blocking=True)  # 异步拷贝到 GPU


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # 构造读索引：(每 token 的槽位号, 每 token 的序列位置)，配合 token 池/页表做 GPU 高级索引取数
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)  # 行索引缓冲（锁页内存）
    offset = 0
    for req in batch.padded_reqs:  # 逐请求填充
        length = req.extend_len  # 该请求 token 数
        mapping_host[offset : offset + length].fill_(req.table_idx)  # 行索引 = 槽位号（重复 length 次）
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)  # (GPU 行索引, 列索引=positions 转 int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # 构造写索引：(每请求槽位号, 新 token 写入位置)；已完成（不可解码）的请求记 -1 表示跳过写入
    mapping_list = [req.table_idx for req in batch.reqs]  # 每请求的槽位号（只含真实请求，不含 dummy）
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)  # 行索引缓冲
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]  # 写入位置 = 当前长度（尾部追加）；-1 = 不写
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)  # 列索引缓冲
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)  # 异步拷贝两个索引到 GPU
