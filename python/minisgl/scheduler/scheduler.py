# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# 类型注解工具；NoReturn 表示永不返回（主循环）；TypeAlias 自定义类型别名
from typing import TYPE_CHECKING, List, NamedTuple, NoReturn, Set, Tuple, TypeAlias

import torch
# Batch/Req：批次与请求对象
from minisgl.core import Batch, Req
# 环境变量开关（如是否关闭重叠调度）
from minisgl.env import ENV
# 消息协议：批量消息、用户请求、中止请求、退出、反解码结果
from minisgl.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    DetokenizeMsg,
    ExitMsg,
    UserMsg,
)
# 工具：日志器与 tokenizer 加载（用于拿 eos_token_id 判断结束）
from minisgl.utils import init_logger, load_tokenizer

# 调度器内部各管理器
from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .io import SchedulerIOMixin
from .prefill import ChunkedReq, PrefillManager
from .table import TableManager

# 仅类型检查导入：引擎侧类型，运行时才真正 import（延迟到 __init__ 避免循环依赖）
if TYPE_CHECKING:
    from minisgl.engine import BatchSamplingArgs, ForwardOutput


# 本模块日志器
logger = init_logger(__name__)

# 二维索引别名：两个等长张量 (行号, 列号)，用于 fancy-index 读写 GPU 张量
Indice2D: TypeAlias = Tuple[torch.Tensor, torch.Tensor]


# For overlap scheduling, we also need to cache some other data to avoid IMA
# 一次 forward 的完整输入快照。重叠调度下，本步调度产生的元数据要留到下步
# 处理结果时使用，因此打包缓存，防止下步覆盖元数据导致非法内存访问（IMA）
class ForwardInput(NamedTuple):
    # 本次执行的批次
    batch: Batch
    # 采样参数（temperature/top-k 等的批量形式）
    sample_args: BatchSamplingArgs
    input_tuple: Indice2D  # (token_mapping, positions)
    # 读索引：(每 token 的槽位行号, 每 token 的位置 id)——取输入 token 与做 RoPE
    write_tuple: Indice2D  # (req_mapping, seq_lens or -1)
    # 写索引：(每请求的槽位行号, 写入位置或 -1)——把新 token 槽位写回页表；
    # -1 表示该请求本轮不落新 KV（如被 padding 或已结束），避免越界写


# 一次"输入+输出"配对：重叠流水线中在两个循环迭代间传递
ForwardData: TypeAlias = "Tuple[ForwardInput, ForwardOutput]"


# 调度器主体：混入 I/O 能力，负责消息处理、批次调度、资源管理与结果回收
class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        # 函数内导入引擎：避免模块加载期的循环依赖
        from minisgl.engine import Engine

        # 创建引擎（模型加载、KV 池、CUDA Graph 等都在这一步完成）
        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        # 调度器使用独立的 CUDA stream：让本步的元数据准备（CPU→GPU 拷贝、页表
        # 写入等）与上一步引擎的计算在硬件上重叠，隐藏 CPU/GPU 交互延迟
        self.device = self.engine.device
        # 调度器自己的流
        self.stream = torch.cuda.Stream(device=self.device)
        # 引擎计算流的上下文管理器（进入后以引擎流为当前流）
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        # 把当前线程的 CUDA 流切到调度器流（之后的拷贝/索引计算都走这条流）
        torch.cuda.set_stream(self.stream)

        # initialize other managers
        # 初始化各管理器：槽位管理器（上限=最大并发请求数，共用引擎页表）
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # KV cache 管理器（页池 + 前缀缓存）
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type
        )
        # decode 管理器（在解码请求集合）
        self.decode_manager = DecodeManager(config.page_size)
        # prefill 管理器（等待队列 + 批次组装）
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )

        # some alias for easy access
        # 上一轮刚结束的请求集合（用于重叠调度下去重，防止同一请求被释放两次）
        self.finished_reqs: Set[Req] = set()
        # 加载 tokenizer 只为获取 eos id（调度器不做文本编解码）
        self.tokenizer = load_tokenizer(config.model_path)
        # 结束符 token id：生成出它即认为请求完成
        self.eos_token_id = self.tokenizer.eos_token_id
        # token 池别名：所有请求的 token id 都存在这里，kernel 直接在 GPU 上读
        self.token_pool = self.table_manager.token_pool
        # prefill 每步的 token 预算（即分块大小）
        self.prefill_budget = config.max_extend_tokens
        # self.config = config

        # Initialize the I/O mixin
        # 初始化 I/O 混入（建立 ZMQ/广播链路；TP 组来自引擎）
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        # 空闲期后台任务：打日志并做 KV cache 完整性自检（纯调试保障，无请求时可安全执行）
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # 是否阻塞等消息：只有既没有待处理的上一批结果、又没有任何可跑请求时才阻塞，
        # 否则忙轮询（重叠调度不能卡在收消息上，否则 GPU 会空转）
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        )
        # 收取并处理本轮所有消息（新请求/中止/退出）
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 调度出下一个要执行的批次（prefill 优先，其次 decode）
        forward_input = self._schedule_next_batch()
        # 本步启动的批次（要留给下一轮迭代去收结果）
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                # 让引擎流等待调度器流上的元数据拷贝完成（跨流依赖）
                self.engine.stream.wait_stream(self.stream)
                # 在引擎流上启动本批 forward（异步发射，不阻塞 CPU）
                ongoing_data = (forward_input, self._forward(forward_input))

        # 处理上一批的结果：追 token、判结束、释放资源、回包给 tokenizer
        self._process_last_data(last_data)
        # 把本批传给下一轮迭代
        return ongoing_data

    def normal_loop(self) -> None:
        # 非重叠模式主循环：调度与结果处理串行，逻辑更简单
        blocking = not (self.prefill_manager.runnable or self.decode_manager.runnable)
        # 无可跑请求时阻塞等消息
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # 调度下一批
        forward_input = self._schedule_next_batch()
        # 本步的输入输出配对
        ongoing_data = None
        if forward_input is not None:
            # 同步执行：forward 返回时结果已就绪
            ongoing_data = (forward_input, self._forward(forward_input))

        # 立即处理本批结果（没有流水线重叠）
        self._process_last_data(ongoing_data)

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        if ENV.DISABLE_OVERLAP_SCHEDULING:
            # 关闭重叠调度：整个循环都在引擎流里跑（调度元数据与计算同流串行）
            with self.engine_stream_ctx:
                # 先让引擎流等调度器流的既有工作完成
                self.engine.stream.wait_stream(self.stream)
                # 无限循环直到收到退出消息
                while True:
                    self.normal_loop()
        else:
            # 重叠调度要求当前流必须是调度器专用流（前面已切好，这里断言防回归）
            assert torch.cuda.current_stream() == self.stream
            # 上一批的输入输出（流水线寄存器），首轮为 None
            data = None
            while True:
                # 每轮：处理上批结果 -> 收消息 -> 启动新批，把新批传给下轮
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        # 等待 GPU 上所有工作完成，确保退出前状态一致
        torch.cuda.synchronize(self.device)
        # 全 rank 屏障：防止某 rank 先退出导致 NCCL 通信挂死
        self.sync_all_ranks()
        # 释放引擎资源并退出进程
        self.engine.shutdown()

    def _process_last_data(self, last_data: ForwardData | None) -> None:
        if last_data is None:
            # 首轮或上轮没有启动批次：无事可做
            return

        # 解包上批的批次对象与 forward 输出 (input_indices, (cpu_tokens, copy_done))
        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        # 等待 GPU→CPU 的异步拷贝完成，此后读 next_tokens_cpu 才是安全的
        copy_done.synchronize()
        # 本轮要发给 tokenizer 的反解码消息
        reply: List[DetokenizeMsg] = []
        # 本轮结束的请求集合（替换旧的，用于重叠调度去重）
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            # 惰性释放区：区域内释放的页先挂起，出区才回收——
            # 因为下面马上还要启动新批次，立即复用页会与新批次的 kernel 竞争
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # 分块请求本轮不产出 token，跳过
                    continue
                # 取该请求新生成的 token（CPU 标量张量）
                next_token = next_tokens_cpu[i]
                # 追加到请求的 CPU 输入序列，保持完整历史
                req.append_host(next_token.unsqueeze(0))
                # 转成 Python int 用于比较与传输
                next_token = int(next_token.item())
                # 结束条件一：不能再解码（输出长度已达 max_tokens）
                finished = not req.can_decode
                if not req.sampling_params.ignore_eos:
                    # 结束条件二：生成了 eos（除非用户要求忽略 eos）
                    finished |= next_token == self.eos_token_id
                # 组装回包消息（uid + 新 token + 是否结束）
                reply.append(DetokenizeMsg(uid=req.uid, next_token=next_token, finished=finished))

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                # 结束的请求做资源回收；重叠调度下同一请求可能在本批和上一批
                # 都被判结束，用 finished_reqs 集合去重避免二次释放
                if finished and req not in self.finished_reqs:
                    # 移出 decode 运行集合
                    self.decode_manager.remove_req(req)
                    # 释放槽位与 KV 资源
                    self._free_req_resources(req)
                    # 记入本轮结束集合
                    new_finished_reqs.add(req)
                elif batch.is_prefill:  # for prefill, non-chunk req, cache the prefix
                    # 未结束且是 prefill 批：把本请求刚算完的整段输入 KV 存入前缀缓存，
                    # 供后续请求（含它自己的 decode 阶段）复用
                    self.cache_manager.cache_req(req, finished=False)

        # 滚动更新结束集合
        self.finished_reqs = new_finished_reqs
        # 把所有 token 消息发给 tokenizer 进程做反解码
        self.send_result(reply)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            # 批量消息：拆包后逐条递归处理
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            # 退出消息：抛键盘中断走统一的退出清理路径
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            # 用户请求消息
            logger.debug_rank0("Received user msg: %s", msg)
            # 输入长度与模型支持的最大序列长度
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            # 输出上限 = 最大序列长 - 输入长
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                # 输入本身超长：直接丢弃该请求并告警
                return logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
            if msg.sampling_params.max_tokens > max_output_len:
                # 请求的 max_tokens 超出模型能力：截断到上限并告警（不丢弃请求）
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            # 入 prefill 等待队列
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            # 中止请求消息
            logger.debug_rank0("Aborting request %d", msg.uid)
            # 先在 prefill 队列找，再在 decode 运行集合找
            req_to_free = self.prefill_manager.abort_req(msg.uid)
            req_to_free = req_to_free or self.decode_manager.abort_req(msg.uid)
            if req_to_free is not None:
                # 找到了就释放其 KV 与槽位资源
                self._free_req_resources(req_to_free)
        else:
            # 未知消息类型：直接报错，防止静默丢消息
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _free_req_resources(self, req: Req) -> None:
        # 归还请求槽位（页表行）
        self.table_manager.free(req.table_idx)
        # 把请求已产生的 KV 交还缓存管理器（finished=True：尾部页也一并释放）
        self.cache_manager.cache_req(req, finished=True)

    def _prepare_batch(self, batch: Batch) -> ForwardInput:
        # CUDA Graph 回放需要固定批量：不足处用 dummy 请求填充对齐
        self.engine.graph_runner.pad_batch(batch)
        # 为批内所有请求统一做增量页分配（只分配新增部分）
        self.cache_manager.allocate_paged(batch.reqs)
        # 生成每个 token 的位置 id（prefill 是连续区间，decode 是单位置）
        batch.positions = _make_positions(batch, self.device)
        # 构造读索引：每 token -> (槽位行号, 位置)
        input_mapping = _make_input_tuple(batch, self.device)
        # 构造写索引：每请求 -> (槽位行号, 写入位置或 -1)
        write_mapping = _make_write_tuple(batch, self.device)
        # 算出每个新 token 在 KV 池中的落点：page_table[行号, 位置] 即槽位下标
        batch.out_loc = self.engine.page_table[input_mapping]
        # 让注意力后端准备本批元数据（paged KV 的索引、seq lens 等）
        self.engine.attn_backend.prepare_metadata(batch)
        # 打包成 ForwardInput 快照（重叠调度下这些张量本步用完，下步还要读）
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _schedule_next_batch(self) -> ForwardInput | None:
        # TODO: support other policies: e.g. DECODE first
        # 调度策略：prefill 优先（尽快消化新请求），无 prefill 才跑 decode
        batch = (
            self.prefill_manager.schedule_next_batch(self.prefill_budget)
            or self.decode_manager.schedule_next_batch()
        )
        # 有批次才做启动前的准备（页分配/元数据），否则返回 None
        return self._prepare_batch(batch) if batch else None

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        # 解包：批次、采样参数、读索引、写索引
        batch, sample_args, input_mapping, output_mapping = forward_input
        # 按 read 索引从 token 池取本批的输入 token（在 GPU 上完成，免 CPU 拼接）
        batch.input_ids = self.token_pool[input_mapping]
        # 执行一次 forward + 采样，得到下一步 token（GPU 上）
        forward_output = self.engine.forward_batch(batch, sample_args)
        # 把新 token 写回 token 池对应位置（decode 时正好是下一步的输入位）
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        # 用本轮参与请求刷新 decode 集合（淘汰已到输出上限的请求）
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        # 返回输出（含拷贝完成事件，供结果处理时同步）
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    # 本批所有 token 的总数（含 padding 请求）
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    # CPU 锁页内存组装位置数组（支持异步 H2D 拷贝）
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    # 当前填充偏移
    offset = 0
    for req in batch.padded_reqs:
        # 该请求本批的 token 数
        length = req.extend_len
        # 位置 = [cached_len, device_len) 的等差序列：
        # 新 token 的绝对位置从已缓存长度开始编号
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        # 推进偏移
        offset += length
    # 一次性异步拷到 GPU
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # 读索引的行号数组：每 token 记录它属于哪个请求（槽位行）
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    # 填充偏移
    offset = 0
    for req in batch.padded_reqs:
        # 该请求的 token 数
        length = req.extend_len
        # 这段 token 全部属于同一行，整段填充同一 table_idx
        mapping_host[offset : offset + length].fill_(req.table_idx)
        # 推进偏移
        offset += length
    # 行号异步上 GPU；列号直接复用 positions（转 int64 用于索引）
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    # 写索引的行号数组：每请求一行
    mapping_list = [req.table_idx for req in batch.reqs]
    # CPU 锁页内存组装
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    # 写入位置：decode 请求写到 device_len（新 token 的位置）；
    # 非 decode（prefill 未完成/分块请求）写 -1 表示"本步不落新 KV"
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    # CPU 锁页内存组装
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    # 两个数组都异步上 GPU
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
