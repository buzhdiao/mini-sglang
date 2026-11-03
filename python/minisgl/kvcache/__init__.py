from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from typing import TYPE_CHECKING, Protocol  # TYPE_CHECKING：条件导入；Protocol：结构化接口（静态鸭子类型）

from minisgl.utils import Registry  # 通用注册表：按名字注册/查找工厂函数

if TYPE_CHECKING:  # 仅静态类型检查时导入，运行时不执行
    import torch  # torch 仅用于类型注解
    from minisgl.models import ModelConfig  # 模型配置，仅用于类型注解

from .base import (  # 从 base 模块导入 KV Cache 子系统的核心抽象
    BaseCacheHandle,  # 缓存句柄基类（标识一段已缓存的 KV 前缀）
    BaseKVCachePool,  # KV Cache 物理池接口（分配/读写 paged KV 存储）
    BasePrefixCache,  # 前缀缓存管理器接口（匹配/插入/淘汰）
    MatchResult,  # 前缀匹配结果（命中的句柄）
    SizeInfo,  # 缓存容量信息（可淘汰/受保护的 token 数）
)


class CacheManagerCreator(Protocol):
    """前缀缓存管理器工厂的协议：给定设备，返回一个前缀缓存管理器实例"""

    def __call__(self, device: torch.device) -> BasePrefixCache: ...  # 工厂签名


# 前缀缓存管理器注册表：各实现（naive/radix）通过装饰器注册，键为类型名
SUPPORTED_CACHE_MANAGER = Registry[CacheManagerCreator]("Cache Manager")


def create_kvcache_pool(
    model_config: ModelConfig,  # 模型配置（层数、头数、头维度）
    num_pages: int,  # KV Cache 总页数（显存预分配的容量）
    page_size: int,  # 每页 token 数
    dtype: torch.dtype,  # KV 数据类型（如 bfloat16）
    device: torch.device,  # 存放设备
) -> BaseKVCachePool:
    """创建 KV Cache 物理池：按模型形状一次性预分配整块显存。"""
    from .mha_pool import MHAKVCache  # TODO: support other variants (e.g. MLA)  # 延迟导入；目前只实现 MHA/GQA 布局，MLA 等待支持

    return MHAKVCache(  # 实例化 MHA 布局的 KV Cache（2×layers×pages×page_size×heads×dim 的大张量）
        num_kv_heads=model_config.num_kv_heads,  # KV 头数（GQA 模型少于 Q 头数）
        num_pages=num_pages,  # 页数
        page_size=page_size,  # 页大小
        num_layers=model_config.num_layers,  # 模型层数（每层独立 KV）
        head_dim=model_config.head_dim,  # 头维度
        device=device,  # 设备
        dtype=dtype,  # 数据类型
    )


@SUPPORTED_CACHE_MANAGER.register("naive")  # 注册为 "naive" 前缀缓存（无复用能力的基线实现）
def create_naive_cache(device: torch.device):
    from .naive_cache import NaivePrefixCache  # 延迟导入实现

    return NaivePrefixCache(device=device)  # 实例化 naive 前缀缓存


@SUPPORTED_CACHE_MANAGER.register("radix")  # 注册为 "radix" 前缀缓存（radix tree 前缀复用）
def create_radix_cache(device: torch.device):
    from .radix_cache import RadixPrefixCache  # 延迟导入实现

    return RadixPrefixCache(device=device)  # 实例化 radix 前缀缓存


def create_prefix_cache(device: torch.device, type: str) -> BasePrefixCache:
    """按类型名从注册表创建前缀缓存管理器（"naive" 或 "radix"）。"""
    return SUPPORTED_CACHE_MANAGER[type](device)  # 查表调用工厂函数


__all__ = [  # 模块公开接口
    "create_kvcache_pool",  # KV 物理池工厂
    "create_prefix_cache",  # 前缀缓存管理器工厂
    "BaseKVCachePool",  # KV 池抽象基类
    "BaseCacheHandle",  # 缓存句柄基类
    "BasePrefixCache",  # 前缀缓存管理器基类
    "SizeInfo",  # 容量信息
    "MatchResult",  # 匹配结果
    "SUPPORTED_CACHE_MANAGER",  # 管理器注册表
]
