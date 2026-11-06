# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# functools.cache 缓存 AOT 模块加载
import functools
# 类型注解工具
from typing import TYPE_CHECKING

# AOT 编译加载工具
from .utils import load_aot

# 仅类型检查时导入
if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module


# 加载 radix.cpp 编译出的模块（只加载一次）
@functools.cache
def _load_radix_module() -> Module:
    # AOT 加载 radix 的 C++ 实现
    return load_aot("radix", cpp_files=["radix.cpp"])


# 比较 2 个一维 int CPU 张量是否相等，返回 0/1
def fast_compare_key(x: torch.Tensor, y: torch.Tensor) -> int:
    # compare 2 1-D int cpu tensors for equality
    # radix cache 前缀匹配的高频操作：C++ 一次 memcmp 级比较远快于 torch.eq
    return _load_radix_module().fast_compare_key(x, y)
