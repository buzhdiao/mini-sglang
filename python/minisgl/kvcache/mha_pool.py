from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

import torch  # PyTorch：张量存储
from minisgl.distributed import get_tp_info  # 张量并行信息（切分 KV 头数）
from minisgl.utils import div_even  # 均分工具：整除并保证结果为偶数（可允许复制）

from .base import BaseKVCachePool  # KV 物理池抽象基类


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.

    MHA/GQA 布局的 paged KV Cache 物理池：把全部 KV 一次性预分配在单块大张量中。
    内存布局为 [2(K/V), num_layers, num_pages, page_size, local_kv_heads, head_dim]，
    以"页"为分配单位实现 paged attention 所需的零散逻辑、连续物理存储。
    """

    def __init__(
        self,
        num_kv_heads: int,  # 模型总的 KV 头数（GQA 下通常少于 Q 头数）
        num_layers: int,  # 模型层数（每层独立 KV）
        head_dim: int,  # 每头维度
        num_pages: int,  # 预分配的总页数（决定可服务的最大总 token 数）
        page_size: int,  # 每页 token 数
        dtype: torch.dtype,  # KV 数据类型（如 bfloat16）
        device: torch.device,  # 存放设备
    ) -> None:
        tp_info = get_tp_info()  # 获取张量并行信息
        # 本 TP rank 负责的 KV 头数：均分且保持偶数；KV 头数少于 TP 度时允许各 rank 复制全部头
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # 一次性分配整块 KV 显存：dim0=2 对应 K 和 V，比分两张张量少一次 cudaMalloc 且布局统一
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers  # 记录层数（供 num_layers 属性返回）
        self._k_buffer = self._kv_buffer[0]  # K 子张量视图：[layers, pages, page_size, heads, dim]
        self._v_buffer = self._kv_buffer[1]  # V 子张量视图：同上（与 K 共享底层存储，零拷贝）
        self._device = device  # 记录设备
        # token 粒度的扁平形状（pages*page_size 合并为一维），store_kv 写入时把页粒度视图压平成该形状
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        # 返回第 index 层的 K cache 视图：[num_pages, page_size, kv_heads, head_dim]（零拷贝切片）
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        # 返回第 index 层的 V cache 视图，同上
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """把本层新算出的 k/v 按 out_loc 指定的槽位写入 cache（散射写入）。

        out_loc 是调度器为每个新 token 分配的 cache 物理位置（可能乱序、不连续），
        这正是 paged cache"逻辑连续、物理零散"的体现。
        """
        from minisgl.kernel import store_cache  # 延迟导入专用 kernel（高效散射写）

        store_cache(  # 调用 kernel：k[v] 形状 [num_tokens, heads, dim]，按 indices 逐行写入 cache
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),  # 该层 K cache 压平成 token 粒度视图（与 out_loc 的 token 索引对齐）
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),  # 该层 V cache 同上
            indices=out_loc,  # 各 token 的目标物理位置
            k=k,  # 新算出的 K（本批 token）
            v=v,  # 新算出的 V
        )

    @property  # 属性：设备
    def device(self) -> torch.device:
        return self._device  # 返回构造时记录的设备

    @property  # 属性：数据类型
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype  # 直接取底层张量的 dtype（单一来源，不会失同步）

    @property  # 属性：层数
    def num_layers(self) -> int:
        return self._num_layers  # 返回层数
