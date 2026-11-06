# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# ABC/abstractmethod 用于定义抽象基类与抽象方法
from abc import ABC, abstractmethod
# dataclass 用于生成不可变句柄类
from dataclasses import dataclass
# NamedTuple 用于定义带字段名的轻量数据结构
from typing import NamedTuple

import torch


class BaseKVCachePool(ABC):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used.
    """

    # 抽象基类：定义分页式 KV cache 物理池的统一接口
    @abstractmethod
    def k_cache(self, index: int) -> torch.Tensor: ...

    # 取第 index 层的 K 缓存视图（形状 (页数, 页大小, 头数, 头维)），供注意力核使用
    @abstractmethod
    def v_cache(self, index: int) -> torch.Tensor: ...
    # 取第 index 层的 V 缓存视图

    @abstractmethod
    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None: ...
    # 把当前层新算出的 K/V 写入物理池：out_loc 给出每个 token 对应的槽位地址，
    # layer_id 指定写哪一层；这是"计算后回填 KV cache"的关键接口

    @property
    @abstractmethod
    def device(self) -> torch.device: ...
    # 池所在设备

    @property
    @abstractmethod
    def dtype(self) -> torch.dtype: ...
    # KV 元素类型（fp16/bf16 等）

    @property
    @abstractmethod
    def num_layers(self) -> int: ...
    # 模型层数（池按层分配存储）


# 冻结数据类：缓存句柄，代表一段已缓存前缀的引用
@dataclass(frozen=True)
class BaseCacheHandle(ABC):
    # 该句柄覆盖的缓存 token 长度
    cached_len: int

    @abstractmethod
    def get_matched_indices(self) -> torch.Tensor: ...
    # 返回该前缀在物理池中的槽位下标，用于注意力计算时直接索引 KV


# 缓存容量信息：可驱逐部分与受保护（已锁定）部分的 token 数
class SizeInfo(NamedTuple):
    evictable_size: int
    protected_size: int

    @property
    def total_size(self) -> int:
        # 总缓存占用 = 可驱逐 + 受保护
        return self.evictable_size + self.protected_size


# 插入结果：cached_len 为插入前已存在于缓存中的前缀长度（这些槽位应被释放），
# handle 为本次插入得到的缓存句柄
class InsertResult(NamedTuple):
    cached_len: int  # length already in cache before insertion (should be freed)
    handle: BaseCacheHandle  # cache handle for the inserted prefix


# 前缀匹配结果：给出可复用前缀的句柄（后续用 lock_handle 锁定后才能安全使用其 indices）
class MatchResult(NamedTuple):
    cuda_handle: BaseCacheHandle
    # TODO: support HiCache


# 抽象基类：前缀缓存管理器（如 naive/radix 实现），负责逻辑层的命中、加锁与驱逐
class BasePrefixCache(ABC):
    @abstractmethod
    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """
        Lock or unlock a cache handle.
        This operation will not modify the cache, but change the size info only.
        When a handle is locked, it cannot be evicted.
        Handles must be locked before the previously-returned tensor of `match_prefix` is used.
        Otherwise it may be evicted by calling evict.

        Args:
            handle (BaseCacheHandle): The cache handle to lock or unlock.
            unlock (bool): Whether to unlock the handle. Defaults to False.
        """
        # 加锁/解锁一个缓存句柄：只改统计信息不改缓存内容。
        # 锁定的句柄不会被 evict 驱逐；使用 match_prefix 返回的 indices 前必须先加锁，
        # 否则并发 evict 可能把这段前缀的槽位回收导致数据失效

    @abstractmethod
    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """
        Match prefix and return the indices of the matched prefix in the cache.
        This operation will not modify the cache.
        The returned indices is only safe to use when the handle is locked.

        Args:
            input_ids (torch.Tensor): The input ids to match. Shape: (seq_len,)
        Returns:
            MatchResult: The match result containing the cache handles.
        """
        # 前缀匹配：查找请求的 token 序列与缓存中已有前缀的最长公共部分，
        # 返回其槽位下标以便直接复用 KV（跳过重复的前缀预填充计算）。
        # 本操作不修改缓存；返回的 indices 只有在句柄加锁后才可安全使用

    @abstractmethod
    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """
        Insert a new prefix into the cache.
        This operation will modify the cache.
        Args:
            input_ids (torch.Tensor): The input ids to insert. Shape: (seq_len,)
            indices (torch.Tensor): The indices to store the new prefix. Shape: (seq_len,)

        Returns:
            InsertResult: The result of the insertion.
        """
        # 把一条序列（通常是刚算完 KV 的请求）作为新前缀插入缓存：
        # input_ids 为 token 序列，indices 为这些 token 占用的池槽位。
        # 返回 InsertResult，其中 cached_len 是插入前已缓存的部分长度（对应槽位需释放）

    @abstractmethod
    def evict(self, size: int) -> torch.Tensor:
        """
        Evict some prefixes from the cache to free up space.
        This operation will modify the cache.
        Note that evict 0 is always safe and does nothing.
        Note that the actual evict size may be larger than the requested size.
        Args:
            size (int): The size to evict.

        Returns:
            torch.Tensor: The indices evicted. Shape: (evict_size,)
        Raises:
            RuntimeError: If the requested size is larger than the evictable size.
        """
        # 从缓存中驱逐前缀腾出空间（只驱逐未锁定的部分）。
        # evict(0) 恒安全且为空操作；实际驱逐量可能大于请求量（按整条前缀驱逐）。
        # 返回被驱逐的槽位下标，供池层回收；请求量超过可驱逐量时抛 RuntimeError

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache manager and the underlying cache."""
        # 重置缓存管理器及底层池，清空所有前缀

    @property
    @abstractmethod
    def size_info(self) -> SizeInfo:
        """Get the size information of the cache."""
        # 查询缓存容量信息（可驱逐/受保护的 token 数）

    @abstractmethod
    def check_integrity(self) -> None:
        """Check the integrity of the cache. Raise an error if the cache is corrupted."""
        # 完整性自检：缓存结构被破坏时抛错，用于调试
