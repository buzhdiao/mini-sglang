from __future__ import annotations  # 启用 PEP 563 延迟注解求值，支持前向引用类型注解

import functools  # 提供 functools.cache，缓存动态库加载结果
from typing import TYPE_CHECKING  # 仅静态类型检查时导入重类型

from .utils import load_aot  # AOT 预编译库加载工具

if TYPE_CHECKING:  # 仅类型检查时导入，运行时无开销
    import torch  # 张量库，仅用于注解
    from tvm_ffi import Module  # tvm-ffi 动态库模块类型


@functools.cache  # 缓存加载：radix 动态库整个进程只加载/链接一次
def _load_radix_module() -> Module:
    """加载 AOT 预编译的 radix 动态库（纯 C++ 实现，无 CUDA）。"""
    # 加载 csrc 预编译产物 radix.cpp 对应的库
    return load_aot("radix", cpp_files=["radix.cpp"])


def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    """快速比较两个一维整型 CPU 张量是否相等（radix cache 前缀匹配用）。

    compare 2 1-D int cpu tensors for equality（原有注释）：
    相比 Python 端逐元素比较，C++ 实现避免多次张量往返开销。
    """
    # 直接转发到 C++ 实现；返回 1/0（或相等性结果），用于 radix tree 插入/查找时比较 token key
    return _load_radix_module().fast_compare_key(x, y)
