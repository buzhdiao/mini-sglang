from __future__ import annotations  # 启用延迟注解求值，注解中的类型可前向引用

from datetime import timedelta  # 用于把秒数转换为进程组超时所需的 timedelta 对象
from typing import Any, Dict, NamedTuple, Tuple  # 类型注解工具：Any/Dict/NamedTuple/Tuple

import torch  # PyTorch 主库：设备管理、张量、CUDA Graph、分布式等
from minisgl.attention import create_attention_backend  # 注意力后端工厂：按名字创建对应实现
from minisgl.core import Batch, Context, Req, set_global_ctx  # 核心数据结构：批次、全局上下文、请求
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info  # 分布式初始化/销毁与 TP 信息设置
from minisgl.kvcache import create_kvcache_pool  # KV Cache 显存池工厂：按页数/页大小一次性预分配
from minisgl.layers import set_rope_device  # 设置 RoPE 旋转位置编码运行的设备（全局生效）
from minisgl.models import create_model, load_weight  # 模型工厂与权重加载：先建骨架再填权重
from minisgl.moe import create_moe_backend  # MoE 后端工厂：按名字创建专家计算实现
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype  # 工具：整除、日志、GPU 架构判断、dtype 上下文

from .config import EngineConfig  # 引擎配置类（本包内相对导入）
from .graph import GraphRunner, get_free_memory, mem_GB  # CUDA Graph 运行器、显存查询与格式化工具
from .sample import BatchSamplingArgs, Sampler  # 采样器与批级采样参数

logger = init_logger(__name__)  # 以模块名为前缀初始化日志器（支持 rank0-only 输出）


class ForwardOutput(NamedTuple):  # NamedTuple：带字段名的不可变元组，轻量返回结构
    next_tokens_gpu: torch.Tensor  # GPU 上的下一步 token id（int32），供调度器立即写入页表
    next_tokens_cpu: torch.Tensor  # 异步拷贝到 CPU 的 token id，供上层（如 detokenizer）消费
    copy_done_event: torch.cuda.Event  # CUDA 事件：等待它即可确保 GPU->CPU 拷贝已完成


class Engine:
    """推理引擎：负责模型/KV Cache/页表/后端的初始化，并执行单批前向 + 采样"""

    def __init__(self, config: EngineConfig):
        assert not torch.cuda.is_initialized()  # 要求构造时 CUDA 尚未初始化，保证流/设备设置不被污染
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)  # 登记 TP rank/size 到全局，供各模块查询
        _adjust_config(config)  # 根据 GPU 架构自动填充 "auto" 后端选择、修正与后端冲突的 page_size

        self.device = torch.device(f"cuda:{config.tp_info.rank}")  # 本进程绑定的 GPU：rank 即本地卡号
        torch.cuda.set_device(self.device)  # 将当前设备设为上述 GPU，后续分配默认在此卡
        torch.manual_seed(42)  # 固定全局随机种子，保证权重初始化/采样可复现
        self.stream = torch.cuda.Stream()  # 创建独立 CUDA 流，与默认流隔离以便精确控制异步
        torch.cuda.set_stream(self.stream)  # 将该流设为当前流，后续 kernel 与拷贝都提交到此流
        self.dtype = config.dtype  # 保存计算精度，供权重/KV Cache 分配时使用
        self.ctx = Context(config.page_size)  # 创建引擎级上下文（页大小、页表、后端等均挂在 ctx 上）
        set_global_ctx(self.ctx)  # 注册为全局上下文，让模型前向等深层代码能拿到页表/后端

        self.tp_cpu_group = self._init_communication(config)  # 初始化进程组，返回 CPU(Gloo) 通信组
        init_free_memory = self._sync_get_memory()[1]  # 记录加载模型前的最大空闲显存（跨 rank 同步）
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")  # 仅 rank0 打印空闲显存

        # ======================= Model initialization ========================
        set_rope_device(self.device)  # 告知 RoPE 层在哪个设备上预计算/执行
        with torch.device("meta"), torch_dtype(config.dtype):  # meta 设备：只建模型骨架不分配真实显存
            self.model = create_model(config.model_config)  # 在 meta 上实例化模型结构（占位参数）
        self.model.load_state_dict(self._load_weight_state_dict(config))  # 将真实（或 dummy）权重填入骨架

        # ======================= KV cache initialization ========================
        self.num_pages = self._determine_num_pages(init_free_memory, config)  # 按剩余显存计算可分配的 KV 页数
        num_tokens = self.num_pages * config.page_size  # KV Cache 总容量（以 token 数计）
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(  # 一次性预分配整块 KV 显存池
            model_config=config.model_config,  # 需要层数/头数/头维度来计算每 token 的 KV 尺寸
            num_pages=self.num_pages + 1,  # +1 for dummy page  # 多留 1 页作为哑页（哨兵页，供 dummy 请求指向）
            page_size=config.page_size,  # 页大小（token/页）
            device=self.device,  # KV Cache 所在 GPU
            dtype=self.dtype,  # KV 元素精度与模型一致
        )

        # ======================= Page table initialization ========================
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        # 页表：行=请求槽位，列=token 位置，值=该 token 的 KV Cache 物理位置；对齐 32 是为了 128 字节对齐
        self.max_seq_len = min(config.max_seq_len, num_tokens)  # 序列长度不能超过模型上限，也不能超过缓存总容量
        aligned_max_seq_len = _align_up_32(self.max_seq_len)  # 向上对齐到 32 的倍数（128B 对齐利于向量访存）
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request  # 多 1 行留给 dummy 请求
            (config.max_running_req + 1, aligned_max_seq_len),  # 形状：请求槽数+1 行 × 对齐后的最大序列长度
            dtype=torch.int32,  # int32 足够表达物理位置，且 GPU 友好
            device=self.device,  # 页表常驻 GPU，供 attention kernel 直接读取
        )

        # ======================= Attention & MoE backend initialization ========================
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(  # 创建注意力后端并挂到 ctx
            config.attention_backend, config.model_config  # 按配置名与模型配置实例化
        )
        if config.model_config.is_moe:  # 仅 MoE 模型需要专家计算后端
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)  # 创建 MoE 后端

        # ======================= Sampler initialization ========================
        self.sampler = Sampler(self.device, config.model_config.vocab_size)  # 采样器：需要设备与词表大小

        post_free_memory = self._sync_get_memory()[0]  # 初始化完成后的最小空闲显存（跨 rank）
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")  # 打印初始化后余量

        # ======================= Graph capture initialization ========================
        self.dummy_req = Req(  # 构造哑请求：CUDA Graph 填充/捕获时用于占位
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),  # 占位输入 id（CPU 上的单元素张量）
            table_idx=config.max_running_req,  # 指向页表预留的最后一行（真实请求永不使用）
            cached_len=0,  # 无已缓存前缀
            output_len=1,  # 只产出 1 个 token，避免越界
            uid=-1,  # 特殊 uid：与任何真实请求区分
            sampling_params=None,  # type: ignore  # 哑请求不参与采样，参数置空
            cache_handle=None,  # type: ignore  # 哑请求不占用 KV 页分配器句柄
        )
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page  # 该行全部指向哑页，读到的 KV 恒为 0
        self.graph_runner = GraphRunner(  # 创建 CUDA Graph 运行器并立即开始捕获各 batch size 的图
            stream=self.stream,  # 捕获与重放使用的 CUDA 流
            device=self.device,  # 捕获所在设备
            model=self.model,  # 被捕获的模型前向
            attn_backend=self.attn_backend,  # 注意力后端（捕获/重放前需准备元数据）
            cuda_graph_bs=config.cuda_graph_bs,  # 显式 batch size 列表（可为 None）
            cuda_graph_max_bs=config.cuda_graph_max_bs,  # 自动模式的最大 batch size
            free_memory=init_free_memory,  # 加载模型前的空闲显存，用于推断可捕获的最大 bs
            max_seq_len=aligned_max_seq_len,  # 对齐后的最大序列长度（供后端预留捕获空间）
            vocab_size=config.model_config.vocab_size,  # 词表大小（logits 缓冲区列数）
            dummy_req=self.dummy_req,  # 哑请求，用于把小批填充到图支持的 bs
        )

    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        """初始化 torch.distributed 进程组；返回 CPU(Gloo) 组，NCCL 通信由 pynccl 接管"""
        if config.tp_info.size == 1 or config.use_pynccl:  # 单卡或启用 pynccl：主后端用 gloo，GPU 通信走 pynccl
            torch.distributed.init_process_group(  # 初始化默认进程组（CPU 后端）
                backend="gloo",  # gloo：CPU 后端，负责控制面与 CUDA Graph 中的小消息
                rank=config.tp_info.rank,  # 本进程 rank
                world_size=config.tp_info.size,  # 总 rank 数
                timeout=timedelta(seconds=config.distributed_timeout),  # 通信超时
                init_method=config.distributed_addr,  # TCP rendezvous 地址
            )
            tp_cpu_group = torch.distributed.group.WORLD  # 默认组即 CPU 组（gloo）
            assert tp_cpu_group is not None  # 确保默认组初始化成功
            max_bytes = (  # 计算单次 allreduce 可能的最大字节数
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize  # 最大 token 数 × 隐藏维 × 元素大小
            )
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)  # 在 CPU 组之上启用 pynccl，预注册最大缓冲区
        else:  # 不用 pynccl：直接以 nccl 为默认后端
            torch.distributed.init_process_group(  # 初始化 NCCL 进程组
                backend="nccl",  # nccl：GPU 集合通信
                rank=config.tp_info.rank,  # 本进程 rank
                world_size=config.tp_info.size,  # 总 rank 数
                timeout=timedelta(seconds=config.distributed_timeout),  # 通信超时
                init_method=config.distributed_addr,  # TCP rendezvous 地址
            )
            tp_cpu_group = torch.distributed.new_group(backend="gloo")  # 额外建一个 CPU 组用于控制面同步
            assert tp_cpu_group is not None  # 确保创建成功
        return tp_cpu_group  # 返回 CPU 组（无论走哪条路径）

    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        """加载权重并整理为 state_dict；支持 dummy 权重模式"""
        if config.use_dummy_weight:  # dummy 模式：不读盘，直接生成同形状随机权重
            return {
                k: torch.randn_like(v, device=self.device)  # 与 meta 参数同形状/同 dtype 的随机张量（真实分配到 GPU）
                for k, v in self.model.state_dict().items()  # 遍历模型全部参数名
            }
        else:  # 正常模式：从磁盘加载 HF 权重
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}  # 加载并转换到目标精度

    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        """计算 KV Cache 可分配的页数：优先用显存估算，允许手动覆盖"""
        new_free_memory = self._sync_get_memory()[1]  # 模型加载后的空闲显存（跨 rank 最大值）
        cache_per_page = (  # 单页 KV Cache 的字节数
            2  # key + value  # K 和 V 两份
            * config.model_config.head_dim  # 每头维度
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)  # TP 切分后的 KV 头数（不足时复制）
            * config.page_size  # 每页 token 数
            * self.dtype.itemsize  # 每元素字节数
            * config.model_config.num_layers  # 层数：每层都有独立 KV
        )
        num_pages = config.num_page_override  # 若用户显式指定页数则直接采用
        if num_pages is None:  # 未指定时按显存自动计算
            model_memory = old_free_memory - new_free_memory  # 模型（及初始化）消耗的显存
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory  # 可供 KV 用的显存 = 总量×比例 - 模型占用
            num_pages = available_memory // cache_per_page  # 能买多少页买多少页（向下取整）
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"  # 至少要有几页，否则无法服务
        num_tokens = num_pages * config.page_size  # 总可服务 token 数
        real_kv_size = num_pages * cache_per_page  # KV Cache 实际占用字节数
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")  # 打印分配概况
        return num_pages  # 返回最终页数

    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""  # 获取各 TP rank 空闲显存的最小/最大值
        torch.cuda.synchronize(self.device)  # 等待本设备全部计算完成，显存状态稳定
        torch.cuda.empty_cache()  # 释放 PyTorch 缓存池，归还显存使统计更真实
        torch.cuda.reset_peak_memory_stats(self.device)  # 重置峰值统计（为后续测量做准备）
        free_memory = get_free_memory(self.device)  # 查询本 rank 当前空闲显存
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)  # 用一次 all_reduce 同时求 min/max：取负后求 MIN 即得 MAX
        torch.distributed.all_reduce(  # 跨 rank 归约（在 CPU 组上，逐元素 MIN）
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        min_free_memory = int(free_mem_tensor[0].item())  # 所有 rank 中的最小空闲显存
        max_free_memory = -int(free_mem_tensor[1].item())  # 取负还原：所有 rank 中的最大空闲显存
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:  # rank 间显存差超过 2 GiB 视为失衡
            logger.error(  # 打印失衡详情
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            raise RuntimeError("Memory across TP ranks are imbalanced")  # 直接失败：失衡会导致按最小值预留的显存被最大值一侧 OOM
        return min_free_memory, max_free_memory  # 返回 (min, max)

    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        """对一个批次执行前向 + 采样，返回下一步 token（GPU/CPU 各一份）及拷贝完成事件"""
        assert torch.cuda.current_stream() == self.stream  # 确保在引擎自己的流上执行（CUDA Graph 重放的前提）
        with self.ctx.forward_batch(batch):  # 进入前向上下文：登记当前批次，模型内部据此取页表/后端
            if self.graph_runner.can_use_cuda_graph(batch):  # decode 批且规模不超过已捕获的最大 bs
                logits = self.graph_runner.replay(batch)  # 重放预捕获的 CUDA Graph（低 CPU 启动开销）
            else:
                logits = self.model.forward()  # 常规 eager 前向（prefill 或超出图范围的大批）

        for req in batch.reqs:  # 遍历批内每个请求
            req.complete_one()  # 推进其输出长度计数（本步 token 已产出）

        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)  # 只对真实（未填充）行采样，并转为 int32
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)  # 发起 GPU->CPU 异步拷贝（不阻塞当前流后续工作）
        copy_done_event = torch.cuda.Event()  # 创建 CUDA 事件用于跟踪拷贝
        copy_done_event.record(self.stream)  # 在当前流记录事件：此点之后 CPU 端可安全读取
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)  # 打包返回三件套

    def shutdown(self) -> None:
        """优雅关闭：先毁图再毁进程组，顺序不能颠倒"""
        self.graph_runner.destroy_cuda_graphs()  # 先销毁 CUDA Graph（NCCL 资源释放前必须做，否则可能挂死）
        torch.distributed.destroy_process_group()  # 销毁默认进程组
        destroy_distributed()  # 清理 minisgl 侧的分布式全局状态（如 pynccl）


def _align_up_32(num: int) -> int:
    return (num + 31) // 32 * 32  # 向上取整到 32 的倍数（128 字节对齐，利于 GPU 向量化访存）


def _adjust_config(config: EngineConfig):
    """按 GPU 架构自动补全 "auto" 后端选择，并修正与后端约束冲突的配置"""

    def override(attr: str, value: Any):  # this is dangerous, use with caution  # 绕过 frozen 限制强行写字段（仅限初始化阶段）
        object.__setattr__(config, attr, value)  # frozen dataclass 会拦截 __setattr__，故用 object 层直写

    if config.attention_backend == "auto":  # 未指定注意力后端：按 GPU 架构自动选择
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")  # Blackwell 用 trtllm；Hopper 用 fa+fi 回退；其余仅 fi
        override("attention_backend", backend)  # 写回选择结果
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")  # 仅 rank0 打印

    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:  # trtllm 内核要求特定页大小
        override("page_size", 64)  # 不满足时强制改为 64
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")  # 提醒用户配置被改

    if config.model_config.is_moe and config.moe_backend == "auto":  # MoE 模型且未指定后端
        override("moe_backend", "fused")  # 默认使用融合 MoE 实现
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")  # 仅 rank0 打印
