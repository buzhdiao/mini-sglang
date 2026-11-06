# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# timedelta 表达分布式超时
from datetime import timedelta
# 类型注解工具
from typing import Any, Dict, NamedTuple, Tuple

# torch 提供 CUDA 管理与张量
import torch
# 按名字创建注意力后端
from minisgl.attention import create_attention_backend
# 核心数据结构：批次/上下文/请求/全局上下文注册
from minisgl.core import Batch, Context, Req, set_global_ctx
# 分布式：注册 TP 信息、启用 PyNCCL、销毁通信
from minisgl.distributed import destroy_distributed, enable_pynccl_distributed, set_tp_info
# 创建 KV cache 池
from minisgl.kvcache import create_kvcache_pool
# 设置 RoPE 计算所在设备
from minisgl.layers import set_rope_device
# 创建模型与加载权重
from minisgl.models import create_model, load_weight
# 创建 MoE 后端
from minisgl.moe import create_moe_backend
# 工具：均分/日志/架构判断/数据类型转换
from minisgl.utils import div_even, init_logger, is_sm90_supported, is_sm100_supported, torch_dtype

# 引擎配置
from .config import EngineConfig
# CUDA Graph 执行器与显存工具
from .graph import GraphRunner, get_free_memory, mem_GB
# 采样器与批量采样参数
from .sample import BatchSamplingArgs, Sampler

# 创建模块级 logger
logger = init_logger(__name__)


# 一次 forward 的产出：GPU token、异步拷回 CPU 的 token 与拷贝完成事件
class ForwardOutput(NamedTuple):
    # GPU 上刚采样出的 next token（int32）
    next_tokens_gpu: torch.Tensor
    # 正在异步拷回 CPU 的 next token（需等事件后再读）
    next_tokens_cpu: torch.Tensor
    # 拷贝完成的 CUDA 事件（调度线程用它做同步）
    copy_done_event: torch.cuda.Event


# 推理引擎：负责模型/KV cache/后端/采样器/CUDA Graph 的初始化与单批 forward
class Engine:
    # 用配置完成全部初始化
    def __init__(self, config: EngineConfig):
        # 必须在 CUDA 初始化前完成设备选择等设置
        assert not torch.cuda.is_initialized()
        # 注册本进程的 TP 信息（全局单例）
        set_tp_info(rank=config.tp_info.rank, size=config.tp_info.size)
        # 按硬件能力修正自动选择的配置项
        _adjust_config(config)

        # 本进程绑定的 GPU 设备（rank 即设备号）
        self.device = torch.device(f"cuda:{config.tp_info.rank}")
        # 把当前线程绑定到该设备
        torch.cuda.set_device(self.device)
        # 固定随机种子，保证 TP 各 rank 权重初始化/采样一致
        torch.manual_seed(42)
        # 创建专用计算流（推理全程在该流上，便于与调度重叠）
        self.stream = torch.cuda.Stream()
        # 切换当前流
        torch.cuda.set_stream(self.stream)
        # 保存数据类型
        self.dtype = config.dtype
        # 创建全局上下文（page_size 等）
        self.ctx = Context(config.page_size)
        # 注册为进程级全局上下文，供内核/后端随时访问
        set_global_ctx(self.ctx)

        # 初始化进程组并拿到 CPU 通信组（用于元数据同步）
        self.tp_cpu_group = self._init_communication(config)
        # 记录初始化通信后的空闲显存（用于估算模型加载量）
        init_free_memory = self._sync_get_memory()[1]
        # 打印加载模型前的空闲显存（仅 rank 0 打印）
        logger.info_rank0(f"Free memory before loading model: {mem_GB(init_free_memory)}")

        # ======================= Model initialization ========================
        # 模型初始化
        # 告知 RoPE 层在哪个设备上计算
        set_rope_device(self.device)
        # 先在 meta 设备上按配置"画"出模型结构（不占显存），并设定 dtype 上下文
        with torch.device("meta"), torch_dtype(config.dtype):
            # 创建模型骨架
            self.model = create_model(config.model_config)
        # 把真实权重装进模型（state_dict 方式一步到位）
        self.model.load_state_dict(self._load_weight_state_dict(config))

        # ======================= KV cache initialization ========================
        # KV cache 初始化
        # 按剩余显存计算可分配的页数
        self.num_pages = self._determine_num_pages(init_free_memory, config)
        # 总可缓存的 token 数 = 页数 x 页大小
        num_tokens = self.num_pages * config.page_size
        # 创建 KV cache 池（K/V 分层存储）
        self.ctx.kv_cache = self.kv_cache = create_kvcache_pool(
            model_config=config.model_config,
            num_pages=self.num_pages + 1,  # +1 for dummy page
            page_size=config.page_size,
            device=self.device,
            dtype=self.dtype,
        )

        # ======================= Page table initialization ========================
        # 页表初始化
        # NOTE: 1. aligned to 128 bytes; 2. store raw locations instead of pages
        # 全局页表说明：1) 每行长度 32 对齐（128 字节）以利向量化读取；2) 存的是 token 槽位地址而非页号
        # 引擎可服务的最长序列受配置与 KV cache 容量双重限制
        self.max_seq_len = min(config.max_seq_len, num_tokens)
        # 把行宽向上对齐到 32 的倍数
        aligned_max_seq_len = _align_up_32(self.max_seq_len)
        # 分配页表：多留一行给 dummy 请求（CUDA Graph 填充用）
        self.ctx.page_table = self.page_table = torch.zeros(  # + 1 for dummy request
            (config.max_running_req + 1, aligned_max_seq_len),
            dtype=torch.int32,
            device=self.device,
        )

        # ======================= Attention & MoE backend initialization ========================
        # 注意力与 MoE 后端初始化
        # 创建注意力后端并登记到全局上下文
        self.ctx.attn_backend = self.attn_backend = create_attention_backend(
            config.attention_backend, config.model_config
        )
        # 仅 MoE 模型需要 MoE 后端
        if config.model_config.is_moe:
            # 创建 MoE 后端并登记
            self.ctx.moe_backend = self.moe_backend = create_moe_backend(config.moe_backend)

        # ======================= Sampler initialization ========================
        # 采样器初始化
        # 创建采样器（按词表大小分配缓冲）
        self.sampler = Sampler(self.device, config.model_config.vocab_size)

        # 统计初始化完成后的空闲显存
        post_free_memory = self._sync_get_memory()[0]
        # 打印（仅 rank 0）
        logger.info_rank0(f"Free memory after initialization: {mem_GB(post_free_memory)}")

        # ======================= Graph capture initialization ========================
        # CUDA Graph 捕获初始化
        # 构造 dummy 请求：用于把批量填充到固定 bs（图回放要求固定形状）；
        # 单 token 输入、占用页表预留的最后一行、无缓存、输出 1 个 token、特殊 uid
        self.dummy_req = Req(
            input_ids=torch.tensor([0], dtype=torch.int32, device="cpu"),
            table_idx=config.max_running_req,
            cached_len=0,
            output_len=1,
            uid=-1,
            sampling_params=None,  # type: ignore
            cache_handle=None,  # type: ignore
        )
        # 把 dummy 行的页表整体指向 dummy 页（永不冲突的保留页）
        self.page_table[self.dummy_req.table_idx].fill_(num_tokens)  # point to dummy page
        # 创建 CUDA Graph 执行器（内部完成各 bs 的捕获）
        self.graph_runner = GraphRunner(
            stream=self.stream,
            device=self.device,
            model=self.model,
            attn_backend=self.attn_backend,
            cuda_graph_bs=config.cuda_graph_bs,
            cuda_graph_max_bs=config.cuda_graph_max_bs,
            free_memory=init_free_memory,
            max_seq_len=aligned_max_seq_len,
            vocab_size=config.model_config.vocab_size,
            dummy_req=self.dummy_req,
        )

    # 初始化 torch.distributed 进程组；按配置选择 NCCL 或 gloo+PyNCCL 组合
    def _init_communication(self, config: EngineConfig) -> torch.distributed.ProcessGroup:
        # 单卡或启用 PyNCCL：主后端用 gloo（CPU），张量通信用自研 PyNCCL（GPU）
        if config.tp_info.size == 1 or config.use_pynccl:
            # 用 gloo 初始化进程组（CPU 组，兼做 bootstrap 与元数据同步）
            torch.distributed.init_process_group(
                backend="gloo",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            # 取全局组作为 TP 的 CPU 组
            tp_cpu_group = torch.distributed.group.WORLD
            # 确保拿到有效组
            assert tp_cpu_group is not None
            # 估算单次 forward 最大的激活张量字节数（PyNCCL 缓冲按此分配）
            max_bytes = (
                config.max_forward_len * config.model_config.hidden_size * self.dtype.itemsize
            )
            # 启用 PyNCCL 通道处理 GPU 集合通信
            enable_pynccl_distributed(config.tp_info, tp_cpu_group, max_bytes)
        else:
            # 传统方案：直接用 NCCL 做 GPU 集合通信
            torch.distributed.init_process_group(
                backend="nccl",
                rank=config.tp_info.rank,
                world_size=config.tp_info.size,
                timeout=timedelta(seconds=config.distributed_timeout),
                init_method=config.distributed_addr,
            )
            # 额外建一个 gloo 组做 CPU 侧同步（日志/控制面）
            tp_cpu_group = torch.distributed.new_group(backend="gloo")
            # 确保拿到有效组
            assert tp_cpu_group is not None
        # 返回 CPU 组供后续同步使用
        return tp_cpu_group

    # 加载权重：真实权重或随机 dummy 权重（性能调试用）
    def _load_weight_state_dict(self, config: EngineConfig) -> Dict[str, torch.Tensor]:
        # dummy 模式：生成与参数同形状的随机权重
        if config.use_dummy_weight:
            # 逐参数随机初始化到本设备
            return {
                k: torch.randn_like(v, device=self.device)
                for k, v in self.model.state_dict().items()
            }
        else:
            # 从磁盘加载并转换成引擎 dtype
            return {k: v.to(self.dtype) for k, v in load_weight(config.model_path, self.device)}

    # 估算 KV cache 可用页数：显存预算减去模型占用后按每页字节数整除
    def _determine_num_pages(self, old_free_memory: int, config: EngineConfig) -> int:
        # 模型加载完成后再测一次空闲显存（各 rank 取最小值）
        new_free_memory = self._sync_get_memory()[1]
        # 每页 KV cache 字节数 = 2(K+V) x head_dim x 本rank KV头数(TP切分, GQA不足时复制) x 页大小 x 元素字节数 x 层数
        cache_per_page = (
            2  # key + value
            * config.model_config.head_dim
            * div_even(config.model_config.num_kv_heads, config.tp_info.size, allow_replicate=True)
            * config.page_size
            * self.dtype.itemsize
            * config.model_config.num_layers
        )
        # 用户显式指定页数则直接采用
        num_pages = config.num_page_override
        # 未指定时按显存自动推算
        if num_pages is None:
            # 模型加载消耗 = 前后空闲显存之差
            model_memory = old_free_memory - new_free_memory
            # 可用预算 = 总显存的 memory_ratio 比例 - 模型占用
            available_memory = int(config.memory_ratio * old_free_memory) - model_memory
            # 整除得到页数
            num_pages = available_memory // cache_per_page

        # 页数必须 >1（另有一页留给 dummy）
        assert num_pages > 1, "Not enough memory for KV cache, try reducing --num-pages"
        # 可缓存 token 总数
        num_tokens = num_pages * config.page_size
        # 实际 KV 显存占用
        real_kv_size = num_pages * cache_per_page
        # 打印分配结果（仅 rank 0）
        logger.info(f"Allocating {num_tokens} tokens for KV cache, K + V = {mem_GB(real_kv_size)}")
        # 返回页数
        return num_pages

    # 同步后取各 TP rank 空闲显存的最小/最大值，并检查显存是否均衡
    def _sync_get_memory(self) -> Tuple[int, int]:
        """Get the min and max free memory across TP ranks."""
        # 等本设备上的计算全部完成
        torch.cuda.synchronize(self.device)
        # 把缓存块还给驱动，测量更准
        torch.cuda.empty_cache()
        # 清零峰值统计
        torch.cuda.reset_peak_memory_stats(self.device)
        # 读本 rank 的空闲显存
        free_memory = get_free_memory(self.device)
        # 打包 [free, -free]，借 MIN 归约同时求最小 free 与最大 free
        free_mem_tensor = torch.tensor([free_memory, -free_memory], device="cpu", dtype=torch.int64)
        # 在 CPU 组上做 MIN 归约（gloo 支持 CPU 张量）
        torch.distributed.all_reduce(
            free_mem_tensor, op=torch.distributed.ReduceOp.MIN, group=self.tp_cpu_group
        )
        # 各 rank 中最小的空闲显存
        min_free_memory = int(free_mem_tensor[0].item())
        # 各 rank 中最大的空闲显存（负号还原）
        max_free_memory = -int(free_mem_tensor[1].item())
        # 相差超过 2GB 视为显存失衡（会导致部分 OOM）
        if max_free_memory - min_free_memory > 2 * 1024 * 1024 * 1024:
            # 打印失衡详情
            logger.error(
                f"Memory across TP ranks are imbalanced:"
                f" min {mem_GB(min_free_memory)}, max {mem_GB(max_free_memory)}"
            )
            # 直接失败，避免后续隐性 OOM
            raise RuntimeError("Memory across TP ranks are imbalanced")

        # 返回 (最小空闲, 最大空闲)
        return min_free_memory, max_free_memory

    # 执行一个批次的 forward：图回放或 eager，随后完成请求推进与采样
    def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
        # 必须在引擎的专用流上调用
        assert torch.cuda.current_stream() == self.stream
        # 登记当前批次到全局上下文（模型 forward 内部读取）
        with self.ctx.forward_batch(batch):
            # decode 批且批量在捕获列表内：走 CUDA Graph 回放（消除 kernel 启动开销）
            if self.graph_runner.can_use_cuda_graph(batch):
                # 回放图得到 logits
                logits = self.graph_runner.replay(batch)
            else:
                # prefill/超大批量：eager 前向
                logits = self.model.forward()

        # forward 成功后推进每个请求的长度计数（新 token 已写入 KV cache）
        for req in batch.reqs:
            # cached_len/device_len 前移一步
            req.complete_one()

        # 只对真实（未填充）部分采样出 next token
        next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
        # 发起异步 D2H 拷贝（调度线程稍后等待事件再读）
        next_tokens_cpu = next_tokens_gpu.to("cpu", non_blocking=True)
        # 创建拷贝完成事件
        copy_done_event = torch.cuda.Event()
        # 在当前流上记录事件（拷贝入队之后）
        copy_done_event.record(self.stream)
        # 返回三件套给调度器
        return ForwardOutput(next_tokens_gpu, next_tokens_cpu, copy_done_event)

    # 关停引擎：释放 CUDA Graph 与分布式资源
    def shutdown(self) -> None:
        # 销毁所有已捕获的图
        self.graph_runner.destroy_cuda_graphs()
        # 销毁进程组
        torch.distributed.destroy_process_group()
        # 清空分布式插件（释放 PyNCCL 通信器）
        destroy_distributed()


# 向上对齐到 32 的倍数（页表行宽按 128 字节对齐利 GPU 向量化访问）
def _align_up_32(num: int) -> int:
    # 经典上取整公式
    return (num + 31) // 32 * 32


# 依据 GPU 架构与其他配置项，修正 config 中的 "auto" 选择
def _adjust_config(config: EngineConfig):
    # frozen dataclass 的受限写入口（仅初始化阶段使用）
    def override(attr: str, value: Any):  # this is dangerous, use with caution
        # 绕过 frozen 限制直接写属性
        object.__setattr__(config, attr, value)

    # 注意力后端未指定时按架构自动选择
    if config.attention_backend == "auto":
        # Blackwell(SM100) 用 trtllm 内核；Hopper(SM90) 用 fa(prefill)+fi(decode) 混合；其余用 fi
        backend = "trtllm" if is_sm100_supported() else ("fa,fi" if is_sm90_supported() else "fi")
        # 写回配置
        override("attention_backend", backend)
        # 打印自动选择结果（仅 rank 0）
        logger.info_rank0(f"Auto-selected attention backend: {config.attention_backend}")

    # trtllm 内核只支持大分页（16/32/64）
    if "trtllm" in config.attention_backend and config.page_size not in [16, 32, 64]:
        # 强制改成 64
        override("page_size", 64)
        # 提示用户页大小被覆盖
        logger.warning_rank0("Page size is overridden to 64 for TRTLLM backend")

    # MoE 模型未指定后端时自动选 fused 实现
    if config.model_config.is_moe and config.moe_backend == "auto":
        # 写回 fused
        override("moe_backend", "fused")
        # 打印自动选择结果
        logger.info_rank0(f"Auto-selected MoE backend: {config.moe_backend}")
