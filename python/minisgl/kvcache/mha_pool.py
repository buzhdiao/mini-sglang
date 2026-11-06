# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

import torch
# TP（张量并行）全局信息：用于确定本进程分到的 KV 头数
from minisgl.distributed import get_tp_info
# 整除工具：把 num_kv_heads 均分到各 TP rank
from minisgl.utils import div_even

from .base import BaseKVCachePool


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    # 标准 MHA/GQA 结构的分页 KV cache 物理池：
    # KV 按 (layer, page, slot, head, dim) 组织，页表由上层前缀缓存管理器维护
    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        # 获取 TP 并行信息（rank 数、rank 号等）
        tp_info = get_tp_info()
        # 把 KV 头均分到各 TP rank；除不尽时允许复制（allow_replicate），保证各 rank 负载均衡
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # 一次性预分配整块 KV 显存：第 0 维 2 表示 K 与 V 合并存储
        # 形状 (2, 层数, 页数, 页大小, 本地KV头数, 头维)，一次性分配避免碎片化
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        # 记录层数
        self._num_layers = num_layers
        # K 缓存：取合并缓冲的第 0 份，视图共享同一块显存
        self._k_buffer = self._kv_buffer[0]
        # V 缓存：取第 1 份
        self._v_buffer = self._kv_buffer[1]
        # 记录设备
        self._device = device
        # 存储形状：把 (页, 槽) 两维摊平成总的槽位数，供写入核按一维下标寻址
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)

    def k_cache(self, index: int) -> torch.Tensor:
        # 返回第 index 层的 K 缓存视图（形状 (页数, 页大小, 头数, 头维)）
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        # 返回第 index 层的 V 缓存视图
        return self._v_buffer[index]

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        # 把当前层新算出的 K/V 散射写入物理池对应槽位
        from minisgl.kernel import store_cache

        # 调用 GPU 写入核：out_loc 是每个 token 的槽位下标（页表展开后的一维地址），
        # 内核按 indices 把 k/v 各行拷入摊平后的缓存，实现 KV 回填
        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    @property
    def device(self) -> torch.device:
        # 返回池所在设备
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        # 返回 KV 元素类型
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        # 返回模型层数
        return self._num_layers
