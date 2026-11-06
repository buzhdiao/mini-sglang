# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# math 用于计算下一个 2 的幂
import math
# dataclass 声明元数据类
from dataclasses import dataclass
# cached_property 缓存只需计算一次的属性
from functools import cached_property
# TYPE_CHECKING 仅类型检查时导入；Dict/List/Literal 用于注解
from typing import TYPE_CHECKING, Dict, List, Literal

# torch 提供张量与 CUDA 事件
import torch
# Batch 是批次对象；get_global_ctx 拿全局上下文
from minisgl.core import Batch, get_global_ctx
# get_tp_info 获取张量并行组信息（头数切分用）
from minisgl.distributed import get_tp_info
# ENV 读取环境变量覆盖项
from minisgl.env import ENV
# div_even 均匀切分头数；init_logger 创建 logger
from minisgl.utils import div_even, init_logger

# 后端抽象基类与元数据基类
from .base import BaseAttnBackend, BaseAttnMetadata
# CUDA Graph 捕获期的公共静态缓冲
from .utils import BaseCaptureData

# 仅类型检查时导入 flashinfer wrapper 与模型配置（运行时延迟导入）
if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from minisgl.models import ModelConfig


# 求不小于 n 的最小的 2 的幂（缓存分配按 2 的幂对齐）
def _next_power_of_2(n: int) -> int:
    # n<=1 时直接返回 1
    if n <= 1:
        return 1
    # 用 log2 上取整后左移，得到下一个 2 的幂
    return 1 << math.ceil(math.log2(n))


# 创建模块级 logger
logger = init_logger(__name__)


# FlashInfer 后端的 CUDA Graph 捕获数据
@dataclass
class FICaptureData(BaseCaptureData):
    # decode 的 last_page_len 恰好可用全 1 张量表示，这里复用 seq_lens 缓冲
    @property
    def one_tensor(self) -> torch.Tensor:
        # 返回 seq_lens（值在 decode 捕获期约定为 1）
        return self.seq_lens

    # FlashInfer page_size=1 时页表就是 token 下标的 ragged 列表
    @property
    def indices(self) -> torch.Tensor:
        # 直接把页表缓冲当作 indices 使用
        return self.page_table


# FlashInfer wrapper 一次运行所需的元数据（CPU/GPU 混合存放）
@dataclass
class FIMetadata(BaseAttnMetadata):
    # fmt: off
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    cu_seqlens_q_gpu:   torch.Tensor  # on gpu
    indices:            torch.Tensor  # on gpu
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    # dataclass 构造后立即校验各张量的存放位置是否符合 wrapper 的要求
    def __post_init__(self) -> None:
        # FlashInfer 路径当前只支持 page_size=1（每 token 一"页"）
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        # 断言 CPU 侧/ GPU 侧张量各就各位，plan() 才能正确消费
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.cu_seqlens_q_gpu.is_cuda
            and self.indices.is_cuda
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )

    # 返回每请求最新 token 在 packed Q 中的下标（decode 时即 0..bs-1）
    def get_last_indices(self, bs: int) -> torch.Tensor:
        # cu_seqlens_q 的右端点减 1 即最后位置
        return self.cu_seqlens_q_gpu[1 : 1 + bs] - 1


# 基于 FlashInfer（fa2 后端）的注意力后端：plan/run 两段式 API
class FlashInferBackend(BaseAttnBackend):
    # 用模型配置初始化 wrapper 与头数切分
    def __init__(self, config: ModelConfig) -> None:
        # 运行时才导入 flashinfer（未安装该后端时不拖累启动）
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        # 保存模型配置
        self.config = config
        # 引用全局 KV cache
        self.kvcache = get_global_ctx().kv_cache
        # KV cache 所在设备
        self.device = self.kvcache.device
        # 分配 128MB 的 float 工作区（FlashInfer plan/run 的临时缓冲）
        self.float_workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        # prefill wrapper：用 tensor core 批量处理变长 Q 的 ragged attention
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        # decode wrapper：每请求仅 1 个新 token 的批量解码
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # NOTE: some hack to reuse the int_workspace_buffer
        # 让 decode wrapper 复用 prefill wrapper 的 int 工作区，省一份显存
        self.int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer
        # 直接替换私有属性实现共享
        self.decode_wrappers._int_workspace_buffer = self.int_workspace_buffer

        # initialize some data members
        # 初始化与 TP 相关的成员
        # 取 TP 并行度
        tp_size = get_tp_info().size
        # 本 rank 分到的 Q 头数（必须整除）
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        # 本 rank 分到的 KV 头数（GQA 下允许复制以凑齐）
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        # 预分配 CPU 侧全 1 pinned 缓冲（last_page_len 用），按需扩容
        self.cached_ones_cpu: torch.Tensor = torch.tensor([], dtype=torch.int32, pin_memory=True)
        # for cuda graph
        # 要捕获的批量列表
        self.capture_bs: List[int] = []
        # 可捕获的最大批量
        self.max_graph_bs = 0
        # 每个捕获批量对应一个专用 CUDA Graph wrapper
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        # 捕获数据（静态缓冲），初始化前为空
        self.capture: FICaptureData | None = None
        # CUDA 事件：用于同步上一次 plan 触发的异步 H2D 拷贝
        self.last_event = torch.cuda.Event()
        # 先记录一次事件
        self.last_event.record()

    # 只在首次遇到该 metadata 时调用 wrapper.plan()（planning 结果被图/复用场景共享）
    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        # 已初始化则直接返回
        if metadata.initialized:
            return

        # 运行时导入，用于 isinstance 判断
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper

        # 标记已初始化，避免重复 plan
        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        # plan() 复用同一块 pinned 主机缓冲并发起异步 H2D 拷贝，
        # 下次 plan 会改写该缓冲，因此先等上一次拷贝完成
        self.last_event.synchronize()
        # decode wrapper 的 plan 签名与 prefill 不同，分支处理
        if isinstance(metadata.wrapper, BatchDecodeWithPagedKVCacheWrapper):
            # decode planning：传入 ragged KV 的 indptr/indices/last_page_len 等布局信息
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            # prefill planning：额外需要 qo_indptr 与因果掩码
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        # 记录本次 plan 触发的异步拷贝完成事件
        self.last_event.record()

    # 返回长度 bs 的全 1 CPU pinned 张量（带 2 的幂扩容缓存）
    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        # 缓存够长则直接切视图，避免重复分配
        if bs <= len(self.cached_ones_cpu):
            # 返回前 bs 个元素
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        # 按 2 的幂扩容，减少反复再分配
        next_len = _next_power_of_2(bs)
        # 新分配全 1 pinned 张量
        self.cached_ones_cpu = torch.ones(next_len, dtype=torch.int32, pin_memory=True)
        # 返回所需长度切片
        return self.cached_ones_cpu[:bs]

    # 一次注意力前向：先 plan（只一次），写 KV，再用 wrapper.run 执行
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # FlashInfer page_size=1：把分页缓存摊平成 (tokens, 1, heads, dim) 视图
        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            # view 不拷贝数据，仅改变形状解释
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        # 取准备好的元数据
        metadata = batch.attn_metadata
        # 确认类型
        assert isinstance(metadata, FIMetadata)
        # 确保 plan 已完成（每份 metadata 只做一次）
        self._initialize_metadata_once(metadata)
        # 把本层新 k/v 写入 KV cache 的指定槽位
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        # 取本层的 K/V 分页缓存
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        # 按 page=1 摊平两个缓存
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))
        # 执行 wrapper 中已规划的 attention kernel
        return metadata.wrapper.run(q=q, paged_kv_cache=kv_cache)

    # 根据批次内容构造 FlashInfer 元数据（plan 所需的布局信息）
    def prepare_metadata(self, batch: Batch) -> None:
        # 取填充后的请求列表
        reqs = batch.padded_reqs

        # 填充后的批量大小
        padded_size = len(reqs)
        # 每请求本次 Q 长度（decode 为 1，padding 为 0）
        seqlens_q = [req.extend_len for req in reqs]
        # 每请求 KV 总长度
        seqlens_k = [req.device_len for req in reqs]
        # 每请求缓存命中长度
        cached_lens = [req.cached_len for req in reqs]
        # 本批最长 Q 长度
        max_seqlen_q = max(seqlens_q)
        # CPU pinned int32 参数，便于异步 H2D
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        # 后端设备
        device = self.device
        # 每请求 KV 长度（CPU）
        seq_len_cpu = torch.tensor(seqlens_k, **CPU_KWARGS)
        # KV cumulative 长度（CPU）：[0, k1, k1+k2, ...]
        cu_seqlens_k_cpu = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        if max_seqlen_q == 1:  # decode with all extend_len = 1
            # decode：Q cumulative 恒为 [0..bs]
            cu_seqlens_q_cpu = torch.arange(0, padded_size + 1, **CPU_KWARGS)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # 无缓存命中：Q 与 K 等长，直接复用
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:  # normal extend prefill, with partial cache hit
            # 部分命中：按新增长度重新累计
            cu_seqlens_q_cpu = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)

        # 全局页表（token 粒度）
        page_table = get_global_ctx().page_table
        # 构造元数据并挂到批次：CPU/GPU 双份 cumulative 长度、ragged 索引、头数切分与 wrapper 选择
        batch.attn_metadata = FIMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=cu_seqlens_q_cpu.to(device, non_blocking=True),
            indices=torch.cat([page_table[req.table_idx, : req.device_len] for req in reqs]),
            last_page_len_cpu=self._get_ones_cpu(padded_size),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_len_cpu,
            dtype=self.kvcache.dtype,
            wrapper=self.decode_wrappers if batch.is_decode else self.prefill_wrapper,
        )

    # CUDA Graph 捕获初始化：分配静态缓冲并把页表压平成一维索引
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # 只允许初始化一次
        assert self.capture is None, "Capture already initialized."
        # 最大捕获批量
        max_bs = max(bs_list)
        # 创建捕获缓冲（页表按 token 数宽，因为 page_size=1）
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        # 把二维页表压平成一维，作为 FlashInfer 的 ragged indices 缓冲（见行尾注释）
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        # 记录最大图批量
        self.max_graph_bs = max_bs
        # 保存捕获数据
        self.capture = capture
        # 保存（升序）捕获批量列表
        self.capture_bs = sorted(bs_list)

    # 决定 decode 是否使用 tensor core 版 kernel（只需确定一次）
    @cached_property
    def use_tensor_cores(self) -> bool:
        # 允许环境变量强制覆盖
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            # 记录覆盖信息
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            # 使用覆盖值
            return overriden_value
        # 计算 GQA 比例（Q 头 / KV 头）
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        # GQA>=4 时 tensor core 版本更快，自动启用
        return GQA >= 4

    # 进入某 bs 的捕获：创建该批量的专用 CUDA Graph wrapper 并完成 plan
    def prepare_for_capture(self, batch: Batch) -> None:
        # 运行时导入 CUDA Graph 专用 wrapper
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        # 当前捕获批量
        bs = batch.size
        # 必须在捕获列表、尚未创建过 wrapper、且捕获数据已初始化
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        # 取捕获数据
        capture = self.capture
        # 为该 bs 创建 wrapper：预绑定固定地址的 indptr/indices/last_page_len 缓冲
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        # 强制使用 fa2 后端（与常规路径一致）
        self.graph_wrappers[bs]._backend = "fa2"
        # 共享 int 工作区，避免重复显存分配
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        # 先按正常路径准备一次元数据
        self.prepare_metadata(batch)
        # 取刚生成的元数据
        metadata = batch.attn_metadata
        # 确认类型
        assert isinstance(metadata, FIMetadata)
        # 把 wrapper 换成图专用 wrapper（捕获期 kernel 选择会固定进图）
        metadata.wrapper = self.graph_wrappers[bs]
        # 对图 wrapper 执行 plan，固定 kernel 配置
        self._initialize_metadata_once(metadata)

    # 回放前：把 wrapper 换成该 bs 的图 wrapper 并补做 plan（拷贝数据由外层完成）
    def prepare_for_replay(self, batch: Batch) -> None:
        # 取元数据与填充后批量
        metadata, bs = batch.attn_metadata, batch.padded_size
        # 必须是未 plan 过的 FIMetadata
        assert isinstance(metadata, FIMetadata) and not metadata.initialized
        # 该 bs 必须在捕获列表
        assert self.capture is not None and bs in self.capture_bs
        # 绑定该批量对应的图 wrapper
        metadata.wrapper = self.graph_wrappers[bs]
        # 重新 plan（在图外执行，规划结果供图内 run 使用）
        self._initialize_metadata_once(metadata)
