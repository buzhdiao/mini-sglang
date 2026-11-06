# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# functools.wraps：装饰器保留原函数元信息
import functools
# contextmanager：封装临时修改/恢复的上下文
from contextlib import contextmanager
# 类型注解工具
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # 仅类型检查导入 torch（延迟真实导入）
    import torch


@contextmanager
def torch_dtype(dtype: torch.dtype):
    import torch  # real import when used
    # 使用时才真正导入 torch

    # 记录当前全局默认 dtype
    old_dtype = torch.get_default_dtype()
    # 临时切换默认 dtype（影响其后创建的张量/参数的精度）
    torch.set_default_dtype(dtype)
    try:
        # 执行 with 体内代码
        yield
    finally:
        # 无论成败都恢复原 dtype（保证不影响其他模块）
        torch.set_default_dtype(old_dtype)


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    import torch.cuda.nvtx as nvtx

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            # 默认使用固定名字
            display_name = name
            if layer_id_field and hasattr(self, layer_id_field):
                # 提供层号字段名时把层号填进名字（如 "layer_{}" -> "layer_3"），
                # Nsight Compute/Systems 里就能按层定位 kernel
                display_name = name.format(getattr(self, layer_id_field))
            with nvtx.range(display_name):
                # 在 NVTX range 内执行原方法（性能分析时形成层级区间）
                return fn(self, *args, **kwargs)

        # 返回包装后的方法
        return wrapper

    # 返回装饰器
    return decorator
