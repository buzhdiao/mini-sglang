# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# dataclass 声明元数据/捕获数据类
from dataclasses import dataclass
# TYPE_CHECKING 仅类型检查时导入；List/Tuple 用于注解
from typing import TYPE_CHECKING, List, Tuple

# torch 提供张量运算
import torch
# Batch 是调度器产出的批次对象；get_global_ctx 拿全局上下文（KV cache 等）
from minisgl.core import Batch, get_global_ctx
# 判断是否为 SM100（Blackwell）架构
from minisgl.utils import is_sm100_supported

# 后端抽象基类与元数据基类
from .base import BaseAttnBackend, BaseAttnMetadata
# CUDA Graph 捕获期的公共静态缓冲
from .utils import BaseCaptureData

# 仅类型检查时导入模型配置
if TYPE_CHECKING:
    from minisgl.models import ModelConfig


# FlashAttention 后端的 CUDA Graph 捕获数据，直接复用基类缓冲即可
@dataclass
class FACaptureData(BaseCaptureData):
    # 无额外字段
    pass


# FlashAttention 后端一次 forward 所需的全部元数据
@dataclass
class FAMetadata(BaseAttnMetadata):
    # KV cumulative 序列长度（ragged 布局 [0, k1, k1+k2, ...]）
    cu_seqlens_k: torch.Tensor
    # Q cumulative 序列长度（decode 时为 [0,1,2,...]）
    cu_seqlens_q: torch.Tensor
    # 每请求的 KV 实际长度（flash_attn_with_kvcache 用）
    cache_seqlens: torch.Tensor
    # 本批最长 KV 长度（kernel 侧优化用）
    max_seqlen_k: int
    # 本批最长 Q 长度（decode 为 1）
    max_seqlen_q: int

    # 本批使用的页表（按页索引，已按 kernel 要求处理）
    page_table: torch.Tensor

    # 返回每请求"最新 token"在 packed Q 中的下标：cu_seqlens_q 的右端点减 1
    def get_last_indices(self, bs: int) -> torch.Tensor:
        # [k1, k1+k2, ...] - 1 即各请求最后一个 token 的位置
        return self.cu_seqlens_q[1 : 1 + bs] - 1


# 基于 sgl-kernel flash_attn_with_kvcache 的注意力后端
class FlashAttentionBackend(BaseAttnBackend):
    # 用模型配置初始化后端
    def __init__(self, config: ModelConfig):
        # 取全局上下文（含 KV cache、page_size、page_table 等）
        ctx = get_global_ctx()
        # 保存模型配置
        self.config = config
        # 引用全局 KV cache 对象（分页存储）
        self.kvcache = ctx.kv_cache
        # KV cache 分页大小
        self.page_size = ctx.page_size
        # CUDA Graph 捕获数据，初始化前为空
        self.capture: FACaptureData | None = None
        # 可捕获的最大批量
        self.max_graph_bs = 0
        # 要捕获的批量列表
        self.capture_bs: List[int] = []
        # softmax 缩放 = 1/sqrt(head_dim)，稳定注意力得分
        self.scale = config.head_dim**-0.5
        # Blackwell(SM100) 用 FA4 内核，其余用 FA3 内核
        self.version = 4 if is_sm100_supported() else 3

    # 一次注意力前向：先把新 k/v 写入分页缓存，再调用 flash_attn_with_kvcache
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # 取调度阶段准备好的元数据
        metadata = batch.attn_metadata
        # 确认类型为 FA 的元数据
        assert isinstance(metadata, FAMetadata)
        # 把本层新产生的 k/v 按 out_loc 指示的槽位写入 KV cache
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        # 调用底层 flash attention 实现：传入本层 K/V 分页缓存、页表、
        # 每请求 KV 长度、Q/KV cumulative 长度、批内最大 Q 长度、softmax 缩放与 FA 内核版本
        return _fa_sgl_impl(
            q=q,
            k_cache=self.kvcache.k_cache(layer_id),
            v_cache=self.kvcache.v_cache(layer_id),
            page_table=metadata.page_table,
            cache_seqlens=metadata.cache_seqlens,
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlen_q,
            softmax_scale=self.scale,
            version=self.version,
        )

    # 在每次 forward 前，根据批次内容构造 FA 所需元数据（CPU 组装 + 异步上卡）
    def prepare_metadata(self, batch: Batch) -> None:
        # 取填充对齐后的请求列表（CUDA Graph 要求固定批量）
        reqs = batch.padded_reqs

        # 填充后的批量大小
        padded_size = len(reqs)
        # 每请求本次要计算的 Q 长度（prefill 为整段，decode 为 1，padding 为 0）
        seqlens_q = [req.extend_len for req in reqs]
        # 每请求 KV 的总长度（含历史缓存 + 本次新增）
        seqlens_k = [req.device_len for req in reqs]
        # 每请求已命中 radix cache 的长度（决定 cu_seqlens_q 形态）
        cached_lens = [req.cached_len for req in reqs]
        # 本批最长 KV 长度
        max_seqlen_k = max(seqlens_k)
        # 本批最长 Q 长度
        max_seqlen_q = max(seqlens_q)
        # CPU 侧 pinned 内存参数：锁页内存可触发异步 H2D 拷贝，隐藏传输延迟
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        # KV cache 所在设备
        device = self.kvcache.device
        # 在 CPU 上组好每请求 KV 长度张量
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)
        # 异步拷到 GPU（pinned 内存支持 non_blocking）
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)
        # KV cumulative 长度：[0, k1, k1+k2, ...]，用于 ragged attention
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)
        # 异步拷到 GPU
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)

        # decode：每请求 Q 长度都是 1，cumulative 就是 0..bs 的等差数列
        if max_seqlen_q == 1:
            # 直接生成 [0,1,...,bs]
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit
            # 无缓存命中时 Q 长度等于 KV 长度，直接复用 cu_seqlens_k
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit
            # 部分命中时 Q 只包含新增部分，需按 extend 长度重新做累计和
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            # 异步拷到 GPU
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)

        # 全局页表（行=请求，列=token 槽位，粒度为 1）
        page_table = get_global_ctx().page_table
        # 全局页表粒度是 1 个 token（见下 NOTE），这里每隔 page_size 取一个即得到"页首地址"序列
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )
        # 页大小 >1 时把 token 级地址除以页大小，换算成页索引
        if self.page_size > 1:
            # 整除下转得到页编号
            new_page_table.div_(self.page_size, rounding_mode="floor")
        # 把元数据挂到批次对象上，forward 时直接读取
        batch.attn_metadata = FAMetadata(
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    # CUDA Graph 捕获初始化：分配固定形状的静态缓冲区
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # 只允许初始化一次
        assert self.capture is None, "Capture already initialized."
        # 取要捕获的最大批量
        max_bs = max(bs_list)
        # 按最大批量/页数创建捕获缓冲（页表按页数 = 序列长度/页大小）
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        # 记录最大图批量
        self.max_graph_bs = max_bs
        # 保存捕获数据
        self.capture = capture
        # 保存（升序）捕获批量列表
        self.capture_bs = sorted(bs_list)

    # 进入某 bs 的捕获：把静态缓冲的对应切片包装成元数据挂上批次
    def prepare_for_capture(self, batch: Batch) -> None:
        # 该 bs 必须在捕获列表里且已完成初始化
        assert (bs := batch.size) in self.capture_bs and self.capture
        # 取捕获数据
        capture = self.capture
        # 用捕获期静态缓冲构造固定地址的元数据：KV/Q cumulative 切到当前 bs、
        # cache_seqlens 占位、最大 KV 长度 = 页数 x 页大小、decode 的 Q 长度恒为 1、页表取前 bs 行
        metadata = FAMetadata(
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],
            cache_seqlens=capture.seq_lens[:bs],
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only
            page_table=capture.page_table[:bs, :],
        )
        # 挂到批次上供 forward 使用
        batch.attn_metadata = metadata

    # 回放前把真实批量的动态数据拷入捕获期静态缓冲（地址不变，图才能复用）
    def prepare_for_replay(self, batch: Batch) -> None:
        # 取当前元数据与填充后批量
        metadata, bs = batch.attn_metadata, batch.padded_size
        # 确认类型
        assert isinstance(metadata, FAMetadata)
        # 必须已初始化且该 bs 在捕获列表
        assert self.capture is not None and bs in self.capture_bs
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)
        # decode 的 cu_seqlens_q 恒为等差数列，无需拷贝
        # 取页表宽度，拷贝时对齐
        table_len = metadata.page_table.size(1)
        # 拷贝 KV cumulative 长度到静态缓冲
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)
        # 拷贝每请求 KV 长度
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)
        # 拷贝页表有效宽度部分
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)


# 对 sgl-kernel flash_attn_with_kvcache 的薄封装：集中传参并给出缺失依赖的提示。
# 可选参数：sm_margin 预留 SM 给通信等任务；window_size 滑动窗口(-1=无限)；
# softcap logit 上限(0=关闭)；num_splits split-KV 切分数(0=自动)；
# pack_gqa 是否把 GQA 的 q 头打包进 CTA(None=自适应)；causal 因果掩码
def _fa_sgl_impl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    version: int,
    sm_margin: int = 0,
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window
    softcap: float = 0.0,  # 0.0 means deactivated
    num_splits: int = 0,  # Can be tuned for speed
    pack_gqa: bool | None = None,  # Can be tuned for speed
    causal: bool = True,
) -> torch.Tensor:
    # 导入失败时给出明确的安装指引
    try:
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:
        # 抛出带修复建议的导入错误
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e

    # 直接调用 flash attention 的分页 KV cache 前向
    return flash_attn_with_kvcache(  # type: ignore
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k_new=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        softmax_scale=softmax_scale,
        sm_margin=sm_margin,
        window_size=window_size,
        softcap=softcap,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        causal=causal,
        ver=version,  # TODO: support FA4 on blackwell
    )
