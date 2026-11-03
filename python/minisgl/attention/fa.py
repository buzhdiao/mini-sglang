from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from dataclasses import dataclass  # dataclass 装饰器，自动生成 __init__/__repr__ 等
from typing import TYPE_CHECKING, List, Tuple  # 类型注解工具：条件导入开关、列表、元组

import torch  # PyTorch：张量运算与设备管理
from minisgl.core import Batch, get_global_ctx  # Batch：调度批次；get_global_ctx：全局上下文（含 KV Cache、页表等）
from minisgl.utils import is_sm100_supported  # 工具函数：检测当前 GPU 是否为 SM100（Blackwell 架构，如 B200）

from .base import BaseAttnBackend, BaseAttnMetadata  # 后端与元数据的抽象基类
from .utils import BaseCaptureData  # CUDA Graph 捕获期静态缓冲区基类

if TYPE_CHECKING:  # 仅静态类型检查时导入，运行时不执行
    from minisgl.models import ModelConfig  # 模型配置，仅用于类型注解


@dataclass  # 数据类装饰器
class FACaptureData(BaseCaptureData):
    """FlashAttention 后端的 CUDA Graph 捕获期数据：字段与基类一致，无额外内容（仅作类型区分）"""

    pass  # 空实现：FA 所需捕获缓冲与 BaseCaptureData 完全相同


@dataclass  # 数据类装饰器
class FAMetadata(BaseAttnMetadata):
    """FlashAttention 一次前向所需的元数据（变长/ragged 批次布局，varlen 形式）。

    FA 的 varlen 接口把整个批次拼成一维序列，靠 cu_seqlens 前缀和切分各请求边界；
    paged KV Cache 则通过 page_table 间接寻址。
    """

    cu_seqlens_k: torch.Tensor  # KV 侧累积长度前缀和（形状 [bs+1]）：第 i 个请求的 KV 区间为 [cu_k[i], cu_k[i+1])
    cu_seqlens_q: torch.Tensor  # Q 侧累积长度前缀和（形状 [bs+1]）：decode 时为 [0,1,...,bs]，prefill 时按 extend_len 累加
    cache_seqlens: torch.Tensor  # 每个请求当前 KV Cache 中的实际序列长度（形状 [bs]），kernel 据此截断注意力范围
    max_seqlen_k: int  # 批内最大 KV 长度：kernel 用于决定 tile 调度网格的上界
    max_seqlen_q: int  # 批内最大 Q 长度：decode 时为 1，prefill 时为最大 extend 长度

    page_table: torch.Tensor  # 页表（形状 [bs, max_pages]）：每行是该请求各逻辑页对应的 KV Cache 物理页号

    def get_last_indices(self, bs: int) -> torch.Tensor:
        # 每个请求最后一个 token 的扁平索引 = 该请求 Q 区间终点 - 1，即 cu_seqlens_q[i+1] - 1
        return self.cu_seqlens_q[1 : 1 + bs] - 1


class FlashAttentionBackend(BaseAttnBackend):
    """基于 FlashAttention（sgl-kernel 的 flash_attn_with_kvcache）的注意力后端。

    支持 prefill（varlen extend）与 decode（单 token）两种模式，统一走同一个
    flash_attn_with_kvcache 接口；同时实现 CUDA Graph 的录制/重放支持。
    """

    def __init__(self, config: ModelConfig):
        ctx = get_global_ctx()  # 获取全局上下文（引擎启动时初始化的单例）
        self.config = config  # 保存模型配置（head_dim 等）
        self.kvcache = ctx.kv_cache  # 引用全局 paged KV Cache 池（读写 K/V 都经由它）
        self.page_size = ctx.page_size  # KV Cache 页大小（每页 token 数），用于页表换算
        self.capture: FACaptureData | None = None  # CUDA Graph 捕获期缓冲区，初始为 None（未初始化）
        self.max_graph_bs = 0  # 记录捕获的最大 batch size（重放时用于校验）
        self.capture_bs: List[int] = []  # 计划捕获的所有 batch size 列表（升序）
        self.scale = config.head_dim**-0.5  # softmax 缩放因子 1/sqrt(d)：防止点积随维度增大而方差爆炸
        self.version = 4 if is_sm100_supported() else 3  # Blackwell（SM100）用 FA4 kernel，其余架构用 FA3

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # 取出 prepare_metadata 阶段构建好的元数据，并做类型收窄（确保是 FA 专属格式）
        metadata = batch.attn_metadata
        assert isinstance(metadata, FAMetadata)
        # 先把本层的 k/v 写入 paged KV Cache：out_loc 是本批各 token 在 cache 中的目标位置索引
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        # 调用 FA kernel：Q 为当前批的新 token，K/V 直接从 paged cache 读取（经由页表）
        return _fa_sgl_impl(
            q=q,  # 查询张量（ragged 扁平布局，靠 cu_seqlens_q 切分）
            k_cache=self.kvcache.k_cache(layer_id),  # 本层的 K cache 切片（形状 [num_pages, page_size, kv_heads, head_dim]）
            v_cache=self.kvcache.v_cache(layer_id),  # 本层的 V cache 切片
            page_table=metadata.page_table,  # 逻辑页 → 物理页映射
            cache_seqlens=metadata.cache_seqlens,  # 各请求实际 KV 长度
            cu_seqlens_q=metadata.cu_seqlens_q,  # Q 侧请求边界（前缀和）
            cu_seqlens_k=metadata.cu_seqlens_k,  # K 侧请求边界（作为 cu_seqlens_k_new 传入，表示新写入部分的边界）
            max_seqlen_q=metadata.max_seqlen_q,  # 最大 Q 长度（调度网格上界）
            softmax_scale=self.scale,  # 1/sqrt(head_dim) 缩放
            version=self.version,  # FA3 或 FA4（按架构自动选择）
        )

    def prepare_metadata(self, batch: Batch) -> None:
        """为一次 prefill/decode 前向构建 FA varlen 元数据。

        三种情形（与 FA 官方 varlen 约定一致）：
        1. decode（每请求 Q 长度=1）：cu_seqlens_q 就是 [0,1,2,...]
        2. prefill 且无缓存命中：Q 与 K 的新增部分完全相同，cu_seqlens_q 直接复用 cu_seqlens_k
        3. prefill 且部分前缀命中缓存：Q 长度（extend_len）短于 K 长度，需单独构造 cu_seqlens_q
        """
        reqs = batch.padded_reqs  # 批内请求列表（CUDA Graph 重放时可能含 padding 请求补齐到固定 bs）

        padded_size = len(reqs)  # 补齐后的批大小（含 padding）
        seqlens_q = [req.extend_len for req in reqs]  # 各请求本轮新增的 Q token 数（decode 恒为 1）
        seqlens_k = [req.device_len for req in reqs]  # 各请求当前在设备上的 KV 总长度（含历史缓存）
        cached_lens = [req.cached_len for req in reqs]  # 各请求命中前缀缓存的长度（这部分无需重算）
        max_seqlen_k = max(seqlens_k)  # 批内最大 KV 长度
        max_seqlen_q = max(seqlens_q)  # 批内最大 Q 长度（decode 时为 1）
        # CPU 侧张量的统一构造参数：int32 + 固定内存（pin_memory），使 H2D 拷贝可异步（non_blocking）
        CPU_KWARGS = {"device": "cpu", "dtype": torch.int32, "pin_memory": True}

        device = self.kvcache.device  # KV Cache 所在的 GPU 设备
        cache_seqlens = torch.tensor(seqlens_k, **CPU_KWARGS)  # 在 CPU 上构造各请求 KV 长度张量
        cache_seqlens = cache_seqlens.to(device, non_blocking=True)  # 异步拷贝到 GPU（pin_memory 支持与计算重叠）
        cu_seqlens_k = torch.tensor([0] + seqlens_k, **CPU_KWARGS).cumsum_(dim=0)  # [0, s0, s0+s1, ...] 原地前缀和，作为 KV 边界
        cu_seqlens_k = cu_seqlens_k.to(device, non_blocking=True)  # 异步拷贝到 GPU

        if max_seqlen_q == 1:  # 情形 1：decode，每请求恰好 1 个 Q token
            # cu_seqlens_q 即 [0,1,2,...,bs]，无需任何长度信息
            cu_seqlens_q = torch.arange(0, padded_size + 1, device=device, dtype=torch.int32)
        elif all(l == 0 for l in cached_lens):  # prefill with no cache hit  # 情形 2：prefill 且全部 token 都是新的
            # 此时新增 Q 与 KV 完全一致，边界复用 cu_seqlens_k，省一次构造与传输
            cu_seqlens_q = cu_seqlens_k
        else:  # normal extend prefill, with partial cache hit  # 情形 3：普通 extend，部分命中前缀缓存
            # 新增 Q 的边界按 extend_len 累加（Q 只包含未缓存的新 token）
            cu_seqlens_q = torch.tensor([0] + seqlens_q, **CPU_KWARGS).cumsum_(dim=0)
            cu_seqlens_q = cu_seqlens_q.to(self.kvcache.device, non_blocking=True)  # 异步拷贝到 GPU

        page_table = get_global_ctx().page_table  # 全局页表：形状 [num_tables, max_len]，以 token 粒度记录每个位置的物理索引
        new_page_table = torch.stack(  # NOTE: global page table treat page_size = 1, we need slice  # 全局页表按 token 粒度（等价 page_size=1）存储，这里需抽取出页首
            # 对每个请求：取该请求页表行中前 max_seqlen_k 个 token、每隔 page_size 取一个，即每页的首 token 索引
            [page_table[req.table_idx, : max_seqlen_k : self.page_size] for req in reqs]
        )  # 堆叠成 [bs, max_pages] 的批级页表
        if self.page_size > 1:
            # 全局页表存的是 token 粒度索引，而 FA 的 paged cache 需要页号，故整除页Size向下取整完成换算（原地除法）
            new_page_table.div_(self.page_size, rounding_mode="floor")
        batch.attn_metadata = FAMetadata(  # 把构建好的元数据挂到 batch 上，供 forward 使用
            cu_seqlens_k=cu_seqlens_k,
            cu_seqlens_q=cu_seqlens_q,
            cache_seqlens=cache_seqlens,
            max_seqlen_k=max_seqlen_k,
            max_seqlen_q=max_seqlen_q,
            page_table=new_page_table,
        )

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        """初始化 CUDA Graph 捕获：按最大 batch size 一次性分配静态缓冲区。"""
        assert self.capture is None, "Capture already initialized."  # 防止重复初始化（只允许初始化一次）
        max_bs = max(bs_list)  # 取最大的 batch size，作为缓冲区行数
        # 创建捕获期静态元数据：页表列数为 max_seq_len/page_size（页粒度），与 forward 所需一致
        capture = FACaptureData.create(max_bs, max_seq_len // self.page_size, self.kvcache.device)
        self.max_graph_bs = max_bs  # 记录最大可重放 bs
        self.capture = capture  # 保存缓冲区，录制/重放时都从这里取地址固定的张量
        self.capture_bs = sorted(bs_list)  # 保存合法 bs 列表（升序），重放时校验用

    def prepare_for_capture(self, batch: Batch) -> None:
        """录制前：基于静态捕获缓冲区构造"占位"元数据（录制时地址即被固定）。"""
        # 校验当前 bs 在计划捕获列表中，且捕获缓冲已初始化（海象运算符顺带把 bs 赋值出来）
        assert (bs := batch.size) in self.capture_bs and self.capture
        capture = self.capture  # 局部引用，简化下方代码
        metadata = FAMetadata(  # 从捕获缓冲区按当前 bs 切出"视图"（非拷贝，地址与缓冲区一致）
            cu_seqlens_k=capture.cu_seqlens_k[: bs + 1],  # KV 边界视图：bs+1 个元素
            cu_seqlens_q=capture.cu_seqlens_q[: bs + 1],  # Q 边界视图（decode 恒为 arange，见基类初始化）
            cache_seqlens=capture.seq_lens[:bs],  # 各请求 KV 长度视图
            # 最大 KV 长度按页表列数 × 页大小估算（缓冲区按最大值分配，故为该 bs 下的上界）
            max_seqlen_k=capture.page_table.size(1) * self.page_size,
            max_seqlen_q=1,  # decode only  # 录制仅用于 decode，Q 长度恒为 1
            page_table=capture.page_table[:bs, :],  # 页表视图：前 bs 行
        )
        batch.attn_metadata = metadata  # 挂到 batch 上，随后录制 forward 时 kernel 拿到的就是这些固定地址

    def prepare_for_replay(self, batch: Batch) -> None:
        """重放前：把真实 batch 的元数据"原地拷贝"进捕获期静态缓冲区（地址不变，内容更新）。"""
        metadata, bs = batch.attn_metadata, batch.padded_size  # 真实元数据与补齐后的批大小
        assert isinstance(metadata, FAMetadata)  # 类型收窄
        assert self.capture is not None and bs in self.capture_bs  # 校验缓冲已初始化且该 bs 可重放
        # cu_seqlens_q is always [0, 1, 2, ..., bs] for decode (i.e. no-op)  # decode 时 Q 边界恒为 arange，与缓冲区初值一致，无需拷贝
        table_len = metadata.page_table.size(1)  # 真实页表的列数（页数）
        self.capture.cu_seqlens_k[: bs + 1].copy_(metadata.cu_seqlens_k)  # 原地覆盖 KV 边界
        self.capture.seq_lens[:bs].copy_(metadata.cache_seqlens)  # 原地覆盖各请求 KV 长度
        self.capture.page_table[:bs, :table_len].copy_(metadata.page_table)  # 原地覆盖页表（仅覆盖有效列）


def _fa_sgl_impl(
    q: torch.Tensor,  # 查询张量（ragged 扁平布局）
    k_cache: torch.Tensor,  # paged K cache（本层切片）
    v_cache: torch.Tensor,  # paged V cache（本层切片）
    page_table: torch.Tensor,  # 页表：逻辑页 → 物理页
    cache_seqlens: torch.Tensor,  # 各请求实际 KV 长度
    cu_seqlens_q: torch.Tensor,  # Q 侧累积边界
    cu_seqlens_k: torch.Tensor,  # K 侧累积边界（新增部分的分界）
    max_seqlen_q: int,  # 最大 Q 长度
    softmax_scale: float,  # softmax 缩放因子
    version: int,  # FA kernel 版本（3 或 4）
    sm_margin: int = 0,  # 预留的 SM 数量（留部分 SM 给其他任务，默认 0 不预留）
    window_size: Tuple[int, int] = (-1, -1),  # -1 means infinite context window  # 滑动窗口注意力窗口（左，右），-1 表示不限制
    softcap: float = 0.0,  # 0.0 means deactivated  # logits 软上限（Gemma 类模型用），0 表示禁用
    num_splits: int = 0,  # Can be tuned for speed  # KV 分块并行数（0 表示自动选择）
    pack_gqa: bool | None = None,  # Can be tuned for speed  # 是否把 GQA 的多个 Q 头打包进一个 block（None 自动决定）
    causal: bool = True,  # 是否使用因果掩码（生成场景恒为 True）
) -> torch.Tensor:
    """对 sgl-kernel 的 flash_attn_with_kvcache 的一层薄封装（参数透传 + 依赖检查）。"""
    try:  # 延迟导入依赖：只在真正调用时才尝试加载 sgl-kernel
        from sgl_kernel.flash_attn import flash_attn_with_kvcache
    except ImportError as e:  # 缺依赖时给出可操作的安装提示
        raise ImportError(
            "sgl_kernel.flash_attn is not found. Please install it with `pip install sgl-kernel`.\n"
            "If you're sure it's correctly installed, try `apt update && apt install libnuma1`."
        ) from e  # 保留原始 ImportError 作为异常链

    return flash_attn_with_kvcache(  # type: ignore  # 调用 FA kernel 做 paged attention（类型签名与 torch stub 不完全一致，故忽略类型检查）
        q=q,  # 当前批的 Q
        k_cache=k_cache,  # paged K cache
        v_cache=v_cache,  # paged V cache
        page_table=page_table,  # 页表
        cache_seqlens=cache_seqlens,  # 实际序列长度
        cu_seqlens_q=cu_seqlens_q,  # Q 边界
        cu_seqlens_k_new=cu_seqlens_k,  # 新写入 KV 的边界（varlen extend 语义）
        max_seqlen_q=max_seqlen_q,  # 最大 Q 长度
        softmax_scale=softmax_scale,  # 缩放因子
        sm_margin=sm_margin,  # 预留 SM
        window_size=window_size,  # 滑动窗口
        softcap=softcap,  # logits 软上限
        num_splits=num_splits,  # KV 分块数
        pack_gqa=pack_gqa,  # GQA 打包开关
        causal=causal,  # 因果掩码
        ver=version,  # TODO: support FA4 on blackwell  # kernel 版本（Blackwell 上 FA4 尚待完善）
    )
