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


# 加载 tensor.cpp 编译出的测试模块（只加载一次）
@functools.cache
def _load_test_tensor_module() -> Module:
    # AOT 加载张量测试的 C++ 实现
    return load_aot("test_tensor", cpp_files=["tensor.cpp"])


# 用 C++ 实现比较两个张量是否一致，返回 0/1（内核单测辅助）
def test_tensor(x: torch.Tensor, y: torch.Tensor) -> int:
    # 调用 C++ 端 test 函数
    return _load_test_tensor_module().test(x, y)
