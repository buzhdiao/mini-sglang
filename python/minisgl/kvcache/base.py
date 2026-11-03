from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from abc import ABC, abstractmethod  # 抽象基类与抽象方法支持
from dataclasses import dataclass  # dataclass 装饰器
from typing import NamedTuple  # 具名元组：不可变的轻量数据结构

import torch  # PyTorch：张量类型注解


class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.

    KV Cache 物理池抽象基类：定义 paged KV 存储的统一接口。
    实现负责一次性预分配显存（形状 [2, layers, pages, page_size, heads, dim]），
    并提供按层切片读写 K/V 的能力。注意力后端（FA/FlashInfer/TRT-LLM）通过该接口访问 cache。
    """

    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...
    # 抽象方法：返回第 index 层的 K cache 视图（形状 [num_pages, page_size, kv_heads, head_dim]）

    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...
    # 抽象方法：返回第 index 层的 V cache 视图，同上

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...
    # 抽象方法：把本层新算出的 k/v 写入 paged cache：out_loc 是每个 token 在 cache 中的目标
    # 位置索引（由调度器分配的空闲槽位），layer_id 指定写入哪一层。

    @property
    @abstractmethod
    def device(self) -> torch.device: ...
    # 抽象属性：cache 所在设备

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...
    # 抽象属性：cache 的数据类型（注意力 kernel 初始化时需要）

    @property
    @abstractmethod
    def num_layers(self) -> int: ...
    # 抽象属性：模型层数（即 cache 的层数维度）


@dataclass(frozen=True)  # 不可变数据类：句柄一经创建其 cached_len 不可被意外修改
class BaseCacheHandle(ABC):
    """缓存句柄抽象基类：代表一段已缓存 KV 前缀的引用，配合前缀缓存管理器使用。

    典型流程：match_prefix 返回句柄 → lock_handle 上锁（防淘汰）→
    通过 get_matched_indices 取回命中的 cache 位置 → 用完后 unlock 归还。
    """

    cached_len: int  # 该句柄对应的已缓存前缀长度（token 数）

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...
    # 抽象方法：返回命中前缀在 KV Cache 中的全部位置索引（int32 张量，形状 [cached_len]），
    # 注意力 kernel 用它从 paged cache 中按乱序物理位置读取 KV。


class SizeInfo(NamedTuple):
    """缓存容量信息（具名元组，单位均为 token 数）"""

    evictable_size: int  # 可淘汰量：未被任何请求引用（ref_count=0）的缓存 token 数
    protected_size: int  # 受保护量：已被请求锁定（ref_count>0）不可淘汰的缓存 token 数

    @property
    def total_size(self) -> int:
        # 缓存总量 = 可淘汰 + 受保护
        return self.evictable_size + self.protected_size


class InsertResult(NamedTuple):
    """insert_prefix 的返回结果"""

    cached_len: int  # length already in cache before insertion (should be freed)  # 插入前已存在于缓存中的前缀长度（这部分本次新分配的位置应释放归还）
    handle: BaseCacheHandle  # cache handle for the inserted prefix  # 指向插入后前缀（含命中部分+新插入部分）的句柄


class MatchResult(NamedTuple):
    """match_prefix 的返回结果"""

    cuda_handle: BaseCacheHandle  # 命中前缀的句柄（GPU 侧缓存）
    # TODO: support HiCache  # 待支持主机侧分层缓存（CPU+GPU 两级）


class BasePrefixCache(ABC):
    """前缀缓存管理器抽象基类：管理 KV Cache 中已计算前缀的复用（匹配/插入/锁定/淘汰）。

    实现方（NaivePrefixCache/RadixPrefixCache）只维护"逻辑到物理"的映射与引用计数，
    物理存储由 BaseKVCachePool 负责。所有操作以 token 为单位。
    """

    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        锁定/解锁一个缓存句柄。
        该操作不修改缓存数据本身，只调整容量统计（evictable/protected）。
        锁定（ref_count 增 1）后的句柄不可被 evict 淘汰；
        使用 match_prefix 返回的索引前必须先锁定，否则可能被并发的 evict 淘汰，
        沿路径上所有祖先节点都会被逐级上锁/解锁（引用计数作用于从命中节点到根的整条链）。

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.  # 要锁定/解锁的句柄
            unlock (bool): Whether to unlock the handle. Defaults to False.  # True 表示解锁（ref_count 减 1）
        """

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        在缓存中匹配 input_ids 的最长已知前缀，返回命中句柄。
        该操作不修改缓存（不改变引用计数/树结构）；
        返回的索引仅在句柄被锁定后使用才是安全的（否则可能被淘汰）。

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)  # 待匹配的 token 序列
        Returns:
            MatchResult: The match result containing the cache handles.  # 匹配结果（含命中句柄与其长度）
        """

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)  # 要插入的 token 序列
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)  # 这些 token 在 KV Cache 中的物理位置

        Returns:
            InsertResult: The result of the insertion.  # 含"插入前已缓存长度"（供上层释放重复分配的槽位）与新句柄

        向缓存插入一个新前缀（token 序列 → 物理位置的映射）。该操作会修改缓存（树结构）。
        """

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.  # 期望释放的 token 数

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)  # 被淘汰 token 的物理位置（可归还槽位池）

        Raises:
            RuntimeError: If the requested size is larger than the evictable size.  # 要求释放量超过可淘汰量时报错

        从缓存淘汰约 size 个 token 以腾出空间。该操作会修改缓存。
        evict(0) 恒安全且不做任何事；实际淘汰量可能略大于请求值（以节点为单位淘汰）。
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""  # 重置缓存管理器（清空全部映射）

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""  # 当前容量信息（可淘汰/受保护 token 数）

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""  # 一致性自检：缓存结构损坏时抛错（调试用）
