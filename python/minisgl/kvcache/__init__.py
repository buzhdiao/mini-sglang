# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# Protocol 用于定义结构化子类型（鸭子类型的静态检查版）
from typing import TYPE_CHECKING, Protocol

# 通用注册表工具，用于按名字注册/查找前缀缓存管理器实现
from minisgl.utils import Registry

# torch 与 ModelConfig 仅在类型检查时导入，避免运行时循环依赖
if TYPE_CHECKING:
    import torch
    from minisgl.models import ModelConfig

# 从 base 模块导出 KV cache 相关的抽象基类与数据结构
from .base import (
    BaseCacheHandle,
    BaseKVCachePool,
    BasePrefixCache,
    MatchResult,
    SizeInfo,
)


# 缓存管理器工厂的协议：接收 device，返回一个前缀缓存管理器实例
class CacheManagerCreator(Protocol):
    def __call__(self, device: torch.device) -> BasePrefixCache: ...


# 已支持的前缀缓存管理器注册表：键为策略名（如 "naive"/"radix"）
SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,
    num_pages: int,
    page_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> BaseKVCachePool:
    # 按模型配置创建底层 KV cache 物理池（分页式 paged KV cache）
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)

    # 目前只实现标准 MHA/GQA 结构的 KV 池：形状 (层数, 页数, 页大小, KV 头数, 头维)
    return MHAKVCache(
        num_kv_heads=model_config.num_kv_heads,
        num_pages=num_pages,
        page_size=page_size,
        num_layers=model_config.num_layers,
        head_dim=model_config.head_dim,
        device=device,
        dtype=dtype,
    )


# 注册 "naive" 策略：朴素前缀缓存，实现简单、便于理解与验证
@SUPPORTED_CACHE_MANAGER.register("naive")
def create_naive_cache(device: torch.device):
    # 延迟导入，避免包加载时的循环依赖
    from .naive_cache import NaivePrefixCache

    return NaivePrefixCache(device=device)


# 注册 "radix" 策略：基数树（radix tree）缓存，可按 token 前缀共享 KV 块
@SUPPORTED_CACHE_MANAGER.register("radix")
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache

    return RadixPrefixCache(device=device)


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    # 依据策略名从注册表实例化对应的前缀缓存管理器
    return SUPPORTED_CACHE_MANAGER[type](device)


# 对外暴露的公共 API
__all__ = [
    "create_kvcache_pool",
    "create_prefix_cache",
    "BaseKVCachePool",
    "BaseCacheHandle",
    "BasePrefixCache",
    "SizeInfo",
    "MatchResult",
    "SUPPORTED_CACHE_MANAGER",
]
