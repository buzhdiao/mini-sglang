from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from dataclasses import dataclass  # dataclass 装饰器
from typing import TYPE_CHECKING, List  # 类型注解工具：条件导入开关、列表

import torch  # PyTorch：张量运算
from minisgl.core import Batch, get_global_ctx  # Batch：调度批次；get_global_ctx：全局上下文（KV Cache、页表）

from .base import BaseAttnBackend, BaseAttnMetadata  # 后端与元数据抽象基类
from .utils import BaseCaptureData  # CUDA Graph 捕获期静态缓冲区基类

if TYPE_CHECKING:  # 仅静态类型检查时导入，运行时不执行
    from minisgl.models import ModelConfig  # 模型配置，仅用于类型注解


@dataclass  # 数据类装饰器
class TRTLLMCaptureData(BaseCaptureData):
    """TRT-LLM 后端的 CUDA Graph 捕获期数据：字段与基类一致，无额外内容（仅作类型区分）"""

    pass  # 空实现：所需捕获缓冲与 BaseCaptureData 完全相同


@dataclass  # 数据类装饰器
class TRTLLMMetadata(BaseAttnMetadata):
    """TRT-LLM 一次前向所需的元数据（字段与 FAMetadata 相同的 varlen + paged 布局）。

    TRT-LLM kernel 同样通过 block_tables（页表）寻址 paged KV Cache，
    通过 cu_seqlens 累积长度切分变长批次。
    """

    cu_seqlens_k: torch.Tensor  # KV 侧累积长度前缀和（形状 [bs+1]），context kernel 需要它切分批次
    cu_seqlens_q: torch.Tensor  # Q 侧累积长度前缀和（形状 [bs+1]），同上
    cache_seqlens: torch.Tensor  # 每个请求当前 KV Cache 实际序列长度（形状 [bs]）
    max_seqlen_k: int  # 批内最大 KV 长度（kernel 网格上界）
    max_seqlen_q: int  # 批内最大 Q 长度（decode 时为 1）

    page_table: torch.Tensor  # 页表（形状 [bs, max_pages]）：逻辑页 → KV Cache 物理页映射，即 TRT-LLM 的 block_tables

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # 每请求最后一个 token 的扁平索引 = cu_seqlens_q[i+1] - 1
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class TensorRTLLMBackend(BaseAttnBackend):
    """基于 FlashInfer 所封装的 TensorRT-LLM decode/context kernel 的注意力后端。

    prefill 走 trtllm_batch_context_with_kv_cache（处理变长上下文），
    decode 走 trtllm_batch_decode_with_kv_cache（单 token 高吞吐解码）。
    与 FA 后端相比：kernel 由 FlashInfer 包一层调用，参数形式不同但元数据构建逻辑一致。
    """

    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()  # 全局上下文
        self.config = config  # 模型配置
        self.kvcache = ctx.kv_cache  # 全局 paged KV Cache 池
        self.page_size = ctx.page_size  # KV Cache 页大小
        self.capture: TRTLLMCaptureData | None = None  # CUDA Graph 捕获期缓冲区（未初始化时为 None）
        self.max_graph_bs = 0  # 最大捕获 bs
        self.capture_bs: List[int] = []  # 计划捕获的 bs 列表
        self.scale = config.head_dim**-0.5  # softmax 缩放因子 1/sqrt(head_dim)
        # 分配 128MB 工作区：TRT-LLM kernel 的临时缓冲（经 FlashInfer 传入）
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # 延迟导入 FlashInfer 封装的 TRT-LLM kernel（避免未选该后端时的导入开销）
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache  # decode kernel
        from flashinfer.prefill import trtllm_batch_context_with_kv_cache  # prefill（context）kernel

        metadata = batch.attn_metadata  # 取出构建好的元数据
        assert isinstance(metadata, TRTLLMMetadata)  # 类型收窄
        # 先把本层 k/v 写入 paged KV Cache（out_loc 为目标位置索引）
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))  # 本层 K/V cache 切片

        if batch.is_prefill:  # prefill：用 context kernel 处理变长新增 token
            return trtllm_batch_context_with_kv_cache(
                query=q,  # 本批新 token 的 Q（ragged 布局）
                kv_cache=kv_cache,  # (K, V) paged cache 元组
                workspace_buffer=self.workspace_buffer,  # kernel 临时工作区
                block_tables=metadata.page_table,  # 页表（TRT-LLM 术语叫 block table）
                seq_lens=metadata.cache_seqlens,  # 各请求实际 KV 长度
                max_q_len=metadata.max_seqlen_q,  # 最大 Q 长度
                max_kv_len=metadata.max_seqlen_k,  # 最大 KV 长度
                bmm1_scale=self.scale,  # QK^T 的缩放（1/sqrt(d)）
                bmm2_scale=1.0,  # PV 的缩放（不缩放）
                cum_seq_lens_q=metadata.cu_seqlens_q,  # Q 侧累积边界（varlen 切分）
                cum_seq_lens_kv=metadata.cu_seqlens_k,  # KV 侧累积边界
                kv_layout="NHD",  # cache 布局：[pages, page_size, heads, dim]
                batch_size=batch.size,  # 真实批大小
                out_dtype=q.dtype,  # 输出数据类型与 Q 一致
            )
        else:  # decode：单 token 生成，用专用 decode kernel
            return trtllm_batch_decode_with_kv_cache(
                query=q,  # 本批的 Q（每请求 1 个 token）
                kv_cache=kv_cache,  # (K, V) paged cache
                workspace_buffer=self.workspace_buffer,  # 临时工作区
                block_tables=metadata.page_table,  # 页表
                seq_lens=metadata.cache_seqlens,  # 各请求实际 KV 长度
                max_seq_len=metadata.max_seqlen_k,  # 最大 KV 长度
                bmm1_scale=self.scale,  # QK^T 缩放
                bmm2_scale=1.0,  # PV 缩放
                kv_layout="NHD",  # cache 布局
                out_dtype=q.dtype,  # 输出类型
            )

    def prepare_metadata(self, batch: Batch) -> None:
        """构建 TRT-LLM 元数据：逻辑与 FA 后端完全一致（cu_seqlens 三分支 + 页表页粒度换算）。"""
        reqs = batch.padded_reqs  # 批内请求（可能含 padding）

        padded_size = len(reqs)  # 补齐后批大小
        seqlens_q = [req.extend_len for req in reqs]  # 各请求本轮新增 token 数
        seqlens_k = [req.device_len for req in reqs]  # 各请求设备上 KV 总长度
        cached_lens = [req.cached_len for req in reqs]  # 各请求命中前缀缓存长度
        max_seqlen_k = max(seqlens_k)  # 批内最大 KV 长度
        max_seqlen_q = max(seqlens_q)  # 批内最大 Q 长度
        # CPU pinned int32 张量统一构造参数（支持异步 H2D）
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device  # GPU 设备
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)  # 各请求 KV 长度（CPU 构造）
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)  # 异步拷到 GPU
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)  # KV 边界前缀和 [0, s0, s0+s1, ...]
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)  # 异步拷到 GPU

        if max_seqlen_q == 1:  # 情形 1：decode（每请求 1 个 Q token）
            # Q 边界即 [0,1,2,...,bs]
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit  # 情形 2：prefill 无缓存命中
            # 新增 Q 与 KV 一致，复用 KV 边界
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit  # 情形 3：部分命中
            # Q 边界按 extend_len 单独累加
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)  # 异步拷到 GPU

        page_table = get_global_ctx().page_table  # 全局 token 粒度页表
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice  # 从 token 粒度页表抽取每页首 token
            # 每请求取前 max_seqlen_k 个 token、步长 page_size 切片 → 每页首元素
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )  # 堆叠为 [bs, max_pages]
        if self.page_size > 1:
            # token 索引整除页大小 → 页号（原地除法，避免额外分配）
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = TRTLLMMetadata(  # 挂载元数据
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获缓冲（按最大 bs / 页数分配静态张量）。"""
        assert self.capture is None, "Capture already initialized."  # 只允许初始化一次
        max_bs = max(bs_list)  # 最大捕获 bs
        capture = TRTLLMCaptureData.create(  # 分配静态缓冲；页表列数按页粒度（token 长度 ÷ 页大小）
            max_bs, max_seq_len // self.page_size, self.kvcache.device
        )
        self.max_graph_bs = max_bs  # 记录最大 bs
        self.capture = capture  # 保存缓冲区
        self.capture_bs = sorted(bs_list)  # 保存合法 bs 列表

    def prepare_for_capture(self, batch: Batch) -> None:
        """录制前：基于捕获期静态缓冲构造占位元数据（录制时地址被固定）。"""
        assert (bs := batch.size) in self.capture_bs and self.capture  # 校验 bs 合法且缓冲就绪
        capture = self.capture  # 局部引用
        metadata = TRTLLMMetadata(  # 从捕获缓冲切视图（地址与缓冲一致）
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],  # KV 边界视图
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],  # Q 边界视图（decode 恒为 arange）
            cache_seqlens=capture.seq_lens[:bs],  # KV 长度视图
            max_seqlen_k=capture.page_table.size(1) * self.page_size,  # 按页表列数 × 页大小取上界
            max_seqlen_q=1,  # decode only  # 录制仅用于 decode
            page_table=capture.page_table[:bs, :],  # 页表视图
        )
        batch.attn_metadata = metadata  # 挂载，录制 forward 时 kernel 使用这些固定地址

    def prepare_for_replay(self, batch: Batch) -> None:
        """重放前：把真实元数据原地拷贝进捕获期静态缓冲（地址不变、内容更新）。"""
        metadata, bs = batch.attn_metadata, batch.padded_size  # 真实元数据与补齐 bs
        assert isinstance(metadata, TRTLLMMetadata)  # 类型收窄
        assert self.capture is not None and bs in self.capture_bs  # 校验缓冲与 bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)  # decode 时 Q 边界恒为 arange，与缓冲初值一致，无需拷贝
        table_len = metadata.page_table.size(1)  # 真实页表列数
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)  # 原地覆盖 KV 边界
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)  # 原地覆盖 KV 长度
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)  # 原地覆盖页表有效列
