import torch

# 引入前缀缓存的抽象基类与相关数据结构
from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo


class NaiveCacheHandle(BaseCacheHandle):
    # 朴素缓存的句柄：朴素策略不做任何前缀复用，句柄恒为空
    empty_tensor: torch.Tensor  # should be set by NaivePrefixCache

    def __init__(self):
        # 初始化为 0 长度缓存（无任何已缓存前缀）
        super().__init__(cached_len=0)

    def get_matched_indices(self) -> torch.Tensor:
        # 没有匹配到任何缓存前缀，返回空张量作为槽位下标
        return self.empty_tensor


class NaivePrefixCache(BasePrefixCache):
    # 朴素前缀缓存：所有接口都退化为"无缓存"语义，
    # 用作正确性基线，便于对比验证 radix 缓存的实现
    def __init__(self, device: torch.device):
        # 记录设备
        self.device = device
        # 预生成一个空的 int32 张量，充当"空槽位下标"
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # 类级别共享该空张量，让所有句柄实例无需各自持有
        NaiveCacheHandle.empty_tensor = self.empty_tensor
        super().__init__()

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        # 无真实缓存，加锁/解锁为空操作
        pass

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        # 永远匹配不到前缀，返回空句柄
        return MatchResult(NaiveCacheHandle())

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        # 不真正缓存任何内容；cached_len=0 表示没有需要释放的重复前缀
        return InsertResult(0, NaiveCacheHandle())

    def evict(self, size: int) -> torch.Tensor:
        # 驱逐 0 个 token 是安全的空操作，直接返回空张量
        if size == 0:
            return self.empty_tensor
        # 非 0 驱逐在朴素策略下无意义，直接报错
        raise NotImplementedError("NaiveCacheManager does not support eviction.")

    def reset(self) -> None:
        # 无状态，重置为空操作
        pass

    @property
    def size_info(self) -> SizeInfo:
        # 缓存占用恒为 0
        return SizeInfo(evictable_size=0, protected_size=0)

    def check_integrity(self) -> None:
        # 无结构可校验，恒通过
        pass
