from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

import functools  # 提供 wraps：装饰器保留原函数的元信息
from contextlib import contextmanager  # 把生成器函数变成 with 可用的上下文管理器
from typing import TYPE_CHECKING  # 仅类型检查时导入的开关

if TYPE_CHECKING:  # 仅静态检查时导入 torch 用于注解；运行时延迟导入以加快启动速度
    import torch


@contextmanager  # 声明为上下文管理器，用法：with torch_dtype(torch.bfloat16): ...
def torch_dtype(dtype: torch.dtype):
    """临时切换 PyTorch 全局默认浮点类型，退出 with 块后自动恢复原值。
    典型用途：模型加载/建权重时指定精度（如 bfloat16），避免逐个张量手动指定 dtype。
    """
    import torch  # real import when used  运行时才真正导入 torch

    old_dtype = torch.get_default_dtype()  # 记住当前默认类型，用于退出时恢复
    torch.set_default_dtype(dtype)  # 切换全局默认类型（影响之后创建的浮点张量）
    try:
        yield  # 挂起，把控制权交给 with 块内的代码
    finally:
        torch.set_default_dtype(old_dtype)  # 无论是否抛异常都恢复原类型，保证不污染全局状态


def nvtx_annotate(name: str, layer_id_field: str | None = None):
    """NVTX 性能分析标注装饰器工厂：给被装饰方法包一层 NVTX range，
    在 Nsight 等分析器时间线上显示 name（可用 {} 占位符嵌入对象属性值，如层号）。
    """
    import torch.cuda.nvtx as nvtx  # 运行时才导入 NVTX 模块（属于 CUDA 工具链）

    def decorator(fn):
        @functools.wraps(fn)  # 保留被装饰函数的 __name__/__doc__ 等元信息
        def wrapper(self, *args, **kwargs):
            display_name = name  # 默认显示名（可能含 {} 占位符）
            if layer_id_field and hasattr(self, layer_id_field):  # 指定了属性名且对象上确实存在该属性
                display_name = name.format(getattr(self, layer_id_field))  # 用属性值（如层号）填充模板占位符
            with nvtx.range(display_name):  # 进入 NVTX range：分析器中标记一段执行区间
                return fn(self, *args, **kwargs)  # 透传调用原方法并返回结果

        return wrapper  # 返回包装后的方法

    return decorator  # 返回装饰器，供 @nvtx_annotate("layer {}") 语法使用
