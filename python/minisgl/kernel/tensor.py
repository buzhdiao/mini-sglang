from __future__ import annotations  # 启用 PEP 563 延迟注解求值，支持前向引用类型注解

import functools  # 提供 functools.cache，缓存动态库加载结果
from typing import TYPE_CHECKING  # 仅静态类型检查时导入重类型

from .utils import load_aot  # AOT 预编译库加载工具

if TYPE_CHECKING:  # 仅类型检查时导入，运行时无开销
    import torch  # 张量库，仅用于注解
    from tvm_ffi import Module  # tvm-ffi 动态库模块类型


@functools.cache  # 缓存加载：test_tensor 动态库整个进程只加载/链接一次
def _load_test_tensor_module() -> Module:
    """加载 AOT 预编译的 test_tensor 动态库（纯 C++ 实现，无 CUDA）。"""
    # 加载 csrc 预编译产物 tensor.cpp 对应的库
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    """测试用张量比较函数：把两个张量传给 C++ 侧处理并返回结果。

    主要用于验证 tvm-ffi 的张量传递通路是否正常（冒烟测试）。
    """
    # 直接转发到 C++ 实现的 test 函数
    return _load_test_tensor_module().test(x, y)
