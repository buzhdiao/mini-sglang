# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# dataclass 声明元数据/捕获数据类
from dataclasses import dataclass
# TYPE_CHECKING 仅类型检查时导入；List 用于注解
from typing import TYPE_CHECKING, List

# torch 提供张量运算
import torch
# Batch 是批次对象；get_global_ctx 拿全局上下文
from minisgl.core import Batch, get_global_ctx

# 后端抽象基类与元数据基类
from .base import BaseAttnBackend, BaseAttnMetadata
# CUDA Graph 捕获期的公共静态缓冲
from .utils import BaseCaptureData

# 仅类型检查时导入模型配置
if TYPE_CHECKING:
    from minisgl.models import ModelConfig


# TRT-LLM 后端的 CUDA Graph 捕获数据，直接复用基类缓冲
@dataclass
class TRTLLMCaptureData(BaseCaptureData):
    # 无额外字段
    pass


# TRT-LLM 内核一次调用所需的元数据
@dataclass
class TRTLLMMetadata(BaseAttnMetadata):
    # KV cumulative 序列长度（ragged 布局）
    cu_seqlens_k: torch.Tensor
    # Q cumulative 序列长度
    cu_seqlens_q: torch.Tensor
    # 每请求 KV 实际长度
    cache_seqlens: torch.Tensor
    # 本批最长 KV 长度
    max_seqlen_k: int
    # 本批最长 Q 长度
    max_seqlen_q: int

    # 分页块表（页索引）
    page_table: torch.Tensor

    # 返回每请求最新 token 在 packed Q 中的下标
    def get_last_indices(self, bs: int) -> torch.Tensor:
        # cu_seqlens_q 右端点减 1
        return self.cu_seqlens_q[1 : 1 + bs] - 1


# 基于 flashinfer trtllm 内核的注意力后端（prefill/decode 分两个入口）
class TensorRTLLMBackend(BaseAttnBackend):
    # 用模型配置初始化
    def __init__(self, config: ModelConfig):
        # 取全局上下文
        ctx = get_global_ctx()
        # 保存模型配置
        self.config = config
        # 引用全局 KV cache
        self.kvcache = ctx.kv_cache
        # KV cache 分页大小
        self.page_size = ctx.page_size
        # CUDA Graph 捕获数据，初始化前为空
        self.capture: TRTLLMCaptureData | None = None
        # 可捕获的最大批量
        self.max_graph_bs = 0
        # 要捕获的批量列表
        self.capture_bs: List[int] = []
        # softmax 缩放 = 1/sqrt(head_dim)
        self.scale = config.head_dim**-0.5
        # 分配 128MB 工作区供 trtllm 内核使用
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
        )

    # 一次注意力前向：prefill 走 context 内核，decode 走批量解码内核
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # 运行时导入 trtllm 的 prefill/decode 内核入口
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache

        # 取准备好的元数据
        metadata = batch.attn_metadata
        # 确认类型
        assert isinstance(metadata, TRTLLMMetadata)
        # 把本层新 k/v 写入 KV cache 指定槽位
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        # 取本层 K/V 分页缓存（页布局，供内核直接寻址）
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))

        # 按批次类型选择内核
        if batch.is_prefill:
            # prefill（context）：处理整段新 token 的注意力
            return trtllm_batch_context_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_q_len=metadata.max_seqlen_q,
                max_kv_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                cum_seq_lens_q=metadata.cu_seqlens_q,
                cum_seq_lens_kv=metadata.cu_seqlens_k,
                kv_layout="NHD",
                batch_size=batch.size,
                out_dtype=q.dtype,
            )
        else:
            # decode：每请求只算 1 个新 token 对全部历史的注意力
            return trtllm_batch_decode_with_kv_cache(
                query=q,
                kv_cache=kv_cache,
                workspace_buffer=self.workspace_buffer,
                block_tables=metadata.page_table,
                seq_lens=metadata.cache_seqlens,
                max_seq_len=metadata.max_seqlen_k,
                bmm1_scale=self.scale,
                bmm2_scale=1.0,
                kv_layout="NHD",
                out_dtype=q.dtype,
            )

    # 在 forward 前根据批次内容构造元数据（CPU 组装 + 异步上卡）
    def prepare_metadata(self, batch: Batch) -> None:
        # 取填充对齐后的请求列表
        reqs = batch.padded_reqs

        # 填充后的批量大小
        padded_size = len(reqs)
        # 每请求本次 Q 长度
        seqlens_q = [req.extend_len for req in reqs]
        # 每请求 KV 总长度
        seqlens_k = [req.device_len for req in reqs]
        # 每请求缓存命中长度
        cached_lens = [req.cached_len for req in reqs]
        # 本批最长 KV 长度
        max_seqlen_k = max(seqlens_k)
        # 本批最长 Q 长度
        max_seqlen_q = max(seqlens_q)
        # CPU pinned int32 参数，支持异步 H2D
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        # KV cache 所在设备
        device = self.kvcache.device
        # 每请求 KV 长度张量（CPU 组装）
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        # 异步拷到 GPU
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        # KV cumulative 长度：[0, k1, k1+k2, ...]
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        # 异步拷到 GPU
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        # decode：Q cumulative 恒为 [0..bs]
        if max_seqlen_q == 1:
            # 直接生成等差数列
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # 无缓存命中：Q 长度与 KV 一致，复用 cu_seqlens_k
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            # 部分命中：按新增长度累计
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            # 异步拷到 GPU
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        # 全局页表（token 粒度）
        page_table = get_global_ctx().page_table
        # 每 page_size 取一个地址得到"页首"序列，再堆叠成批的块表（见行尾注释）
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        # token 地址除以页大小得到页索引
        if self.page_size > 1:
            # 整除下转
            new_page_table.div_(self.page_size, rounding_mode="floor")
        # 元数据挂到批次
        batch.attn_metadata = TRTLLMMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    # CUDA Graph 捕获初始化：按页数分配静态缓冲
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # 只允许初始化一次
        assert self.capture is None, "Capture already initialized."
        # 最大捕获批量
        max_bs = max(bs_list)
        # 页表宽度按页数 = 序列长度/页大小
        capture = TRTLLMCaptureData.create(
            max_bs, max_seq_len // self.page_size, self.kvcache.device
        )
        # 记录最大图批量
        self.max_graph_bs = max_bs
        # 保存捕获数据
        self.capture = capture
        # 保存（升序）捕获批量列表
        self.capture_bs = sorted(bs_list)

    # 进入某 bs 的捕获：用静态缓冲切片构造固定地址元数据
    def prepare_for_capture(self, batch: Batch) -> None:
        # 该 bs 必须在捕获列表且已初始化
        assert (bs := batch.size) in self.capture_bs and self.capture
        # 取捕获数据
        capture = self.capture
        # 固定地址的元数据（捕获期占位，回放前写入真值）：cumulative 切到当前 bs、
        # cache_seqlens 占位、最大 KV 长度 = 页数 x 页大小、decode 的 Q 长度恒为 1、页表取前 bs 行
        metadata = TRTLLMMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        # 挂到批次
        batch.attn_metadata = metadata

    # 回放前把真实动态数据拷入静态缓冲（保持张量地址不变）
    def prepare_for_replay(self, batch: Batch) -> None:
        # 取元数据与填充后批量
        metadata, bs = batch.attn_metadata, batch.padded_size
        # 确认类型
        assert isinstance(metadata, TRTLLMMetadata)
        # 该 bs 必须在捕获列表
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        # decode 的 Q cumulative 恒定，无需拷贝
        # 页表有效宽度
        table_len = metadata.page_table.size(1)
        # 拷贝 KV cumulative 长度
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        # 拷贝每请求 KV 长度
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        # 拷贝页表有效部分
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)
