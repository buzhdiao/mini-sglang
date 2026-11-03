from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

import math  # 数学库：_next_power_of_2 中用 log2/ceil
from dataclasses import dataclass  # dataclass 装饰器
from functools import cached_property  # 缓存属性装饰器：首次访问计算后缓存结果
from typing import TYPE_CHECKING, Dict, List, Literal  # 类型注解工具

import torch  # PyTorch：张量与 CUDA 事件
from minisgl.core import Batch, get_global_ctx  # Batch：调度批次；get_global_ctx：全局上下文
from minisgl.distributed import get_tp_info  # 张量并行信息（TP size，用于切分注意力头）
from minisgl.env import ENV  # 环境变量配置项（可覆盖 FlashInfer 的 tensor core 开关）
from minisgl.utils import div_even, init_logger  # div_even：整除并保证偶数（可允许复制）；init_logger：日志器

from .base import BaseAttnBackend, BaseAttnMetadata  # 后端与元数据抽象基类
from .utils import BaseCaptureData  # CUDA Graph 捕获期静态缓冲区基类

if TYPE_CHECKING:  # 仅静态类型检查时导入（flashinfer 较重，运行时在 __init__ 中才真正导入）
    from flashinfer import (  # FlashInfer 的三种 wrapper 类型，仅用于类型注解
        BatchDecodeWithPagedKVCacheWrapper,  # decode 专用 wrapper（单 token）
        BatchPrefillWithPagedKVCacheWrapper,  # prefill 专用 wrapper（变长扩展）
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,  # 支持 CUDA Graph 的 decode wrapper
    )
    from minisgl.models import ModelConfig  # 模型配置，仅用于类型注解


def _next_power_of_2(n: int) -> int:
    """返回不小于 n 的最小 2 的幂（n<=1 时返回 1）。用于按 2 的幂对齐分配、便于复用。"""
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))  # 1 << ceil(log2(n))：把 n 向上取整到最近的 2 的幂


logger = init_logger(__name__)  # 创建本模块 logger


@dataclass  # 数据类装饰器
class FICaptureData(BaseCaptureData):
    """FlashInfer 后端的 CUDA Graph 捕获期数据：在基类字段上补充两个语义别名。"""

    @property  # 别名属性：FlashInfer 的 decode plan 需要的 seq_lens 即基类的 seq_lens
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens  # last_page_len 缓冲区复用：page_size=1 时每"页"长度恒为 1，故可用全 1 的 seq_lens 顶替

    @property  # 别名属性：FlashInfer 的 ragged 索引是一维的，复用基类的二维页表缓冲
    def indices(self) -> torch.Tensor:
        return self.page_table  # 捕获期把 page_table 压平成一维（见 init_capture_graph），作为 ragged KV 索引


@dataclass  # 数据类装饰器
class FIMetadata(BaseAttnMetadata):
    """FlashInfer 一次前向所需的元数据。

    FlashInfer 采用 plan/run 两段式：plan 根据元数据预计算调度信息，run 执行 kernel。
    plan 需要 CPU 侧张量（内部做 H2D 异步拷贝），故这里同时持有 CPU/GPU 两个副本。
    """

    # fmt: off  # 告诉格式化工具（black）不要重排以下字段，保持对齐的可读性
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu  # Q 边界前缀和（CPU 副本，供 plan 使用）
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu  # KV 边界前缀和（CPU 副本，供 plan 使用）
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu  # Q 边界前缀和（GPU 副本，用于取 last indices）
    indices:            torch.Tensor  # on gpu  # ragged KV 索引：批内所有请求的 cache 位置拼接
    last_page_len_cpu:  torch.Tensor  # on cpu  # 每请求最后一页的长度（page_size=1 时恒为 1）
    num_qo_heads:       int  # 本 TP rank 的 Q 头数
    num_kv_heads:       int  # 本 TP rank 的 KV 头数
    head_dim:           int  # 每头维度
    page_size:          Literal[1] # currently only support page_size=1  # 页大小：当前 FlashInfer 路径只支持 page_size=1（token 粒度索引）
    pos_encoding_mode:  str  # 位置编码模式（"NONE"：RoPE 已在模型层处理，kernel 不再叠加）
    seq_lens_cpu:       torch.Tensor  # on cpu  # 各请求 KV 长度（CPU 副本，供 plan 使用）
    dtype:              torch.dtype  # KV Cache 数据类型
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper  # 执行本次前向的 FlashInfer wrapper（按 prefill/decode 选择）
    initialized:        bool = False  # plan 是否已执行过（plan 只需一次；CUDA Graph 重放时不再重复 plan）
    # fmt: on  # 恢复格式化

    def __post_init__(self) -> None:
        """dataclass 生成 __init__ 后自动调用的钩子：校验字段约束。"""
        assert self.page_size == 1, "Currently only page_size=1 is supported."  # 只支持 token 粒度页
        assert (  # 校验各张量所在设备：plan 输入须在 CPU，索引/Q 边界须在 GPU
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # 每请求最后一个 token 的扁平索引 = cu_seqlens_q[i+1] - 1（用 GPU 副本，避免 CPU-GPU 同步）
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    """基于 FlashInfer 的注意力后端：prefill/decode 分别用不同 wrapper，plan+run 两段执行。

    特点：元数据大量使用 CPU pinned 张量（plan 内部异步 H2D）；通过 hack 共享 int workspace
    以省显存；CUDA Graph 用 CUDAGraphBatchDecodeWithPagedKVCacheWrapper 支持重放。
    """

    def __init__(self, config: ModelConfig) -> None:
        # 延迟导入 FlashInfer wrapper：初始化后端时才加载（避免未选该后端时的导入开销）
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config  # 保存模型配置
        self.kvcache = get_global_ctx().kv_cache  # 全局 KV Cache 池引用
        self.device = self.kvcache.device  # 计算设备（与 KV Cache 一致）
        # 分配 128MB 的 float 工作区：FlashInfer kernel 的临时缓冲（prefill/decode 共用）
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        # prefill wrapper：NHD 布局表示 cache 形状为 [pages, page_size, heads, dim]；显式选 fa2 后端
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead  # FlashInfer 的 fa3 路径实测较慢，退回 fa2
        )
        # decode wrapper：按 GQA 比例决定是否走 tensor core（use_tensor_cores 为缓存属性）
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead  # 同上，使用 fa2
        )

        # NOTE: some hack to reuse the int_workspace_buffer  # hack：让 decode wrapper 复用 prefill wrapper 的 int 工作区
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer  # 取出 prefill 的私有 int 工作区（访问内部属性）
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer  # 直接替换 decode 的同名属性，两者共享同一块缓冲，省一份显存

        # initialize some data members  # 初始化与 TP 相关的注意力头数
        tp_size = get_tp_info().size  # 张量并行度
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)  # 本 rank 分到的 Q 头数（均分且保持偶数，利于 kernel 对齐）
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)  # 本 rank 的 KV 头数（KV 头少于 TP 度时允许复制）

        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)  # CPU 侧全 1 张量缓存（供 last_page_len 复用），初始为空
        # for cuda graph  # CUDA Graph 相关状态
        self.capture_bs: List[int] = []  # 计划捕获的 bs 列表
        self.max_graph_bs = 0  # 最大捕获 bs
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}  # bs → 各 bs 专属的 graph decode wrapper
        self.capture: FICaptureData | None = None  # 捕获期静态缓冲区
        self.last_event = torch.cuda.Event()  # CUDA 事件：用于跟踪上一次 plan 的异步 H2D 拷贝是否完成
        self.last_event.record()  # 先记录一次初始事件，保证首次 synchronize 安全

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        """对元数据执行一次 FlashInfer plan（幂等：已初始化则跳过）。

        plan 根据序列长度/页表预计算 kernel 调度信息，绑定到 wrapper 上；
        之后每层 attention 只需调用 run（每层 KV cache 不同但调度相同）。
        """
        if metadata.initialized:  # 已 plan 过（例如后续层复用同一批元数据），直接返回
            return

        from flashinfer import BatchDecodeWithPagedKVCacheWrapper  # 导入 decode wrapper 类型，仅用于 isinstance 判断

        metadata.initialized = True  # 先置标志，避免下方 plan 过程中重入
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        # plan 内部复用同一块 pinned 主机中转缓冲并发起异步 H2D 拷贝；
        # 若上一次 plan 的拷贝未完成就再次 plan，会改写在途数据，故先等上一次事件完成
        self.last_event.synchronize()  # CPU 阻塞等待上次 plan 的拷贝落地
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):  # decode 路径：plan 参数更简单
            metadata.wrapper.plan(  # 预计算 decode kernel 调度
                indptr=metadata.cu_seqlens_k_cpu,  # 各请求 KV 边界（CPU）
                indices=metadata.indices,  # ragged KV 索引（GPU）
                last_page_len=metadata.last_page_len_cpu,  # 最后一页长度（page_size=1 恒为 1）
                num_qo_heads=metadata.num_qo_heads,  # Q 头数
                num_kv_heads=metadata.num_kv_heads,  # KV 头数
                head_dim=metadata.head_dim,  # 头维度
                page_size=metadata.page_size,  # 页大小（恒为 1）
                pos_encoding_mode=metadata.pos_encoding_mode,  # 位置编码模式
                seq_lens=metadata.seq_lens_cpu,  # 各请求 KV 长度（CPU）
                data_type=metadata.dtype,  # （旧版参数位）数据类型
                q_data_type=metadata.dtype,  # Q 数据类型
                kv_data_type=metadata.dtype,  # KV 数据类型
                non_blocking=True,  # 内部 H2D 拷贝不阻塞（由 last_event 机制保证安全）
            )
        else:  # prefill 路径：plan 参数更丰富（需 Q 边界与因果掩码）
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,  # Q 边界前缀和（CPU）
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,  # 每请求页索引区间的边界（CPU）
                paged_kv_indices=metadata.indices,  # ragged 页索引（GPU）
                paged_kv_last_page_len=metadata.last_page_len_cpu,  # 最后一页长度
                num_qo_heads=metadata.num_qo_heads,  # Q 头数
                num_kv_heads=metadata.num_kv_heads,  # KV 头数
                head_dim_qk=metadata.head_dim,  # QK 头维度
                page_size=metadata.page_size,  # 页大小
                pos_encoding_mode=metadata.pos_encoding_mode,  # 位置编码模式
                seq_lens=metadata.seq_lens_cpu,  # 各请求 KV 长度
                q_data_type=metadata.dtype,  # Q 数据类型
                kv_data_type=metadata.dtype,  # KV 数据类型
                non_blocking=True,  # 异步 H2D
                causal=True,  # prefill 使用因果掩码
            )
        self.last_event.record()  # 记录本次 plan 的拷贝事件，供下一次 plan 前 synchronize

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        """获取长度为 bs 的 CPU 全 1 int32 张量（带缓存，按 2 的幂扩容）。"""
        if bs <= len(self.cached_ones_cpu):  # 缓存够用时直接切片返回（零分配）
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2  # 扩容到下一个 2 的幂，减少反复重建
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)  # 新建 pinned 全 1 张量
        return self.cached_ones_cpu[:bs]  # 返回所需长度的切片

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1  # 内部工具：把 [pages, page_size, H, D] 压平成 token 粒度 [tokens, 1, H, D]
            # page_size=1 语义下，FlashInfer 期望每"页"恰好 1 个 token：view 改变形状而非拷贝
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata  # 取出准备好的元数据
        assert isinstance(metadata, FIMetadata)  # 类型收窄
        self._initialize_metadata_once(metadata)  # 确保本批已 plan（每批一次，多层数据复用）
        # 先把本层 k/v 写入 paged KV Cache（out_loc 为目标位置索引）
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))  # 取本层 K/V cache 切片
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))  # 压平成 FlashInfer 期望的 token 粒度布局
        # run：按 plan 好的调度执行 paged attention（q 为本批新 token 的 Q）
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    def prepare_metadata(self, batch: Batch) -> None:
        """构建 FlashInfer 元数据（CPU pinned 为主 + 必要的 GPU 副本）。

        cu_seqlens_q 的三分支逻辑与 FA 后端一致：decode / 无缓存命中 prefill / 部分 hit 的 extend。
        """
        reqs = batch.padded_reqs  # 批内请求（可能含 padding）

        padded_size = len(reqs)  # 补齐后批大小
        seqlens_q = [req.extend_len for req in reqs]  # 各请求本轮新增 token 数（decode 恒 1）
        seqlens_k = [req.device_len for req in reqs]  # 各请求设备上 KV 总长度
        cached_lens = [req.cached_len for req in reqs]  # 各请求命中前缀缓存长度
        max_seqlen_q = max(seqlens_q)  # 批内最大 Q 长度
        # CPU pinned int32 张量统一构造参数（供 plan 的 CPU 输入使用）
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.device  # GPU 设备
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)  # 各请求 KV 长度（CPU）
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)  # KV 边界前缀和（CPU）
        if max_seqlen_q == 1:  # decode with all extend_len = 1  # 情形 1：decode
            # Q 边界即 [0,1,2,...,bs]
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit  # 情形 2：prefill 无缓存命中
            # Q 与 KV 新增部分一致，直接复用 KV 边界
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit  # 情形 3：部分命中
            # Q 边界按 extend_len 累加
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        page_table = get_global_ctx().page_table  # 全局 token 粒度页表
        batch.attn_metadata = FIMetadata(  # 构建并挂载元数据
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),  # Q 边界的 GPU 副本（用于采样取 last indices）
            # ragged 索引：把批内每个请求的页表前 device_len 个位置直接拼接（page_size=1 时即 token 索引）
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),  # 最后一页长度：page_size=1 恒为 1（取缓存的全 1 张量切片）
            num_qo_heads=self.qo_head_local,  # 本 rank Q 头数
            num_kv_heads=self.kv_head_local,  # 本 rank KV 头数
            head_dim=self.config.head_dim,  # 头维度
            page_size=1,  # 固定 token 粒度页
            pos_encoding_mode="NONE",  # RoPE 已在模型内完成，kernel 不再处理
            seq_lens_cpu=seq_len_cpu,  # 各请求 KV 长度（CPU）
            dtype=self.kvcache.dtype,  # KV cache 的数据类型
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,  # 按 decode/prefill 选 wrapper
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获缓冲：页表压平成一维以适配 FlashInfer 的 ragged 索引。"""
        assert self.capture is None, "Capture already initialized."  # 只允许初始化一次
        max_bs = max(bs_list)  # 最大捕获 bs
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)  # 按最大值分配静态缓冲（此处传 token 长度而非页数）
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices  # 二维页表压平成一维：page_size=1 时索引本来就是逐 token 的
        self.max_graph_bs = max_bs  # 记录最大 bs
        self.capture = capture  # 保存缓冲区
        self.capture_bs = sorted(bs_list)  # 保存合法 bs 列表

    @cached_property  # 首次访问时计算并缓存：GQA 比例固定，结果不会变
    def use_tensor_cores(self) -> bool:
        """决定 FlashInfer decode 是否使用 tensor core kernel：GQA 比例 >= 4 时收益明显。"""
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:  # 环境变量可强制覆盖（海象运算符取值）
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")  # 记录覆盖行为
            return overriden_value  # 使用用户指定值
        GQA = self.config.num_qo_heads // self.config.num_kv_heads  # GQA 比例 = Q 头数 / KV 头数
        return GQA >= 4  # 比例较大时 tensor core 的批量化收益才划算

    def prepare_for_capture(self, batch: Batch) -> None:
        """录制前准备：为该 bs 创建专属 graph wrapper，并立即执行一次 plan 固化调度。"""
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper  # 延迟导入 graph 专用 wrapper

        bs = batch.size  # 当前真实 bs
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture  # 校验：bs 可录制、未重复、缓冲就绪
        capture = self.capture  # 局部引用
        # 为该 bs 创建 graph wrapper：预绑定捕获期的静态缓冲区（录制后地址不可变）
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],  # indptr 静态缓冲（bs+1）
            indices_buffer=capture.indices,  # ragged 索引静态缓冲（一维）
            last_page_len_buffer=capture.one_tensor[:bs],  # last_page_len 静态缓冲（全 1 切片）
        )
        self.graph_wrappers[bs]._backend = "fa2"  # hack：强制与普通 wrapper 一致使用 fa2 后端
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer  # hack：共享 int 工作区省显存
        self.prepare_metadata(batch)  # 先按普通流程构建元数据（内容即捕获时的占位数据）
        metadata = batch.attn_metadata  # 取出刚构建的元数据
        assert isinstance(metadata, FIMetadata)  # 类型收窄
        metadata.wrapper = self.graph_wrappers[bs]  # 把 wrapper 换成该 bs 的 graph wrapper（缓冲地址一致，语义等价）
        self._initialize_metadata_once(metadata)  # 立即 plan：graph wrapper 在录制时需要固化调度信息

    def prepare_for_replay(self, batch: Batch) -> None:
        """重放前准备：把 wrapper 切换为该 bs 的 graph wrapper 并执行一次 plan。

        与 FA/TRT-LLM 手动 copy_ 进捕获缓冲不同：graph wrapper 的 plan 会把传入的
        新元数据（metadata 中新鲜的 CPU/GPU 张量）自动拷入其预绑定的静态缓冲，
        随后录制的 run 便从这些固定地址读到本次批的真实数据。
        """
        metadata, bs = batch.attn_metadata, batch.padded_size  # 重放批的元数据与补齐 bs
        assert isinstance(metadata, FIMetadata) and not metadata.initialized  # 元数据必须是"未 plan"的重放元数据
        assert self.capture is not None and bs in self.capture_bs  # 校验捕获缓冲就绪且 bs 合法
        metadata.wrapper = self.graph_wrappers[bs]  # 绑定该 bs 的 graph wrapper（其内部缓冲即录制时的固定地址）
        self._initialize_metadata_once(metadata)  # 执行 plan：内部把新元数据拷入静态缓冲并固化调度
