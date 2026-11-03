import torch  # PyTorch：空张量构造

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo  # 前缀缓存子系统的接口与数据结构


class NaiveCacheHandle(BaseCacheHandle):
    """Naive 缓存的句柄：永远表示"空匹配"（cached_len=0），仅返回共享的空张量"""

    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache  # 类属性：共享空张量，由 NaivePrefixCache 在初始化时注入

    def __init__(self):
        super().__init__(cached_len=0)  # naive 方案无前缀复用，命中长度恒为 0

    def get_matched_indices(self) -> torch.Tensor:
        return self.empty_tensor  # 返回共享的空张量（无任何命中索引）


class NaivePrefixCache(BasePrefixCache):
    """朴素前缀缓存：不做任何前缀复用的基线实现（所有匹配/插入均为 no-op）。

    用途：作为 radix tree 缓存的对照组（便于测试与性能对比），或在前缀复用被
    显式禁用时充当占位实现。所有接口都满足协议但无实际缓存行为。
    """

    def __init__(self, device: torch.device):
        self.device = device  # 记录设备（接口要求）
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)  # 本实例的空张量（int32 索引类型与其他实现一致）
        NaiveCacheHandle.empty_tensor = self.empty_tensor  # 注入到句柄类属性：所有句柄共享同一个空张量
        super().__init__()  # 调用 ABC 基类初始化

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        pass  # no-op：没有真实缓存，无需引用计数/防淘汰逻辑

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        return MatchResult(NaiveCacheHandle())  # 永远返回"空匹配"句柄（cached_len=0，无索引）

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        return InsertResult(0, NaiveCacheHandle())  # 不真正插入：cached_len=0（上层无需释放任何重复位置），返回空句柄

    def evict(self, size: int) -> torch.Tensor:
        if size == 0:  # 按协议约定 evict(0) 恒安全
            return self.empty_tensor  # 返回空张量
        raise NotImplementedError("NaiveCacheManager does not support eviction.")  # 无缓存可淘汰，非零请求直接报错

    def reset(self) -> None:
        pass  # no-op：无状态可重置

    @property  # 属性：容量信息
    def size_info(self) -> SizeInfo:
        return SizeInfo(evictable_size=0, protected_size=0)  # 恒为 0：不占用任何缓存容量

    def check_integrity(self) -> None:
        pass  # no-op：无结构可校验，恒通过
