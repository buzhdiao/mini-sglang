from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如 torch.Tensor | None）

from typing import TYPE_CHECKING  # TYPE_CHECKING 在运行时为 False，仅类型检查器可见，用于避免不必要的 import 开销

if TYPE_CHECKING:  # 仅在静态类型检查时导入，运行时不导入 torch（本文件延迟导入以加快启动速度）
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """SiLU 门控激活：将输入沿最后一维切成两半，输出 silu(前半) * 后半。

    用于 SwiGLU 类 FFN：x 形状 [tokens, 2*intermediate]，输出 [tokens, intermediate]。
    out 提供时原地写入，避免额外显存分配。
    """
    from flashinfer import silu_and_mul  # 延迟导入：flashinfer 是重量级 CUDA 库，用到时才加载

    return silu_and_mul(x, out=out)  # 调用 flashinfer 的 CUDA kernel 完成融合计算，out 为 None 时返回新张量


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    """GELU 门控激活：将输入沿最后一维切成两半，输出 gelu(前半) * 后半。

    形状约定与 silu_and_mul 相同，用于 GeGLU 类 FFN（如部分 GLM/Qwen 变体）。
    """
    from flashinfer import gelu_and_mul  # 延迟导入 flashinfer kernel，避免模块加载时的导入开销

    return gelu_and_mul(x, out=out)  # 调用 CUDA 融合 kernel：一次访存同时完成激活与逐元素相乘


__all__ = ["silu_and_mul", "gelu_and_mul"]  # 声明模块公开 API，约束 from module import * 的导出范围
