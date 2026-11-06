# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def silu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    # 门控激活融合算子：x 的前半部分经 SiLU 激活后与后半部分逐元素相乘。
    # 即 x=[a|b] -> silu(a)*b，一次 GPU 核完成激活+门控，避免两次显存往返
    from flashinfer import silu_and_mul

    return silu_and_mul(x, out=out)


def gelu_and_mul(x: torch.Tensor, out: torch.Tensor | None = None):
    # 同上的 GELU 版本：gelu(前半)*后半，用于 GELU 门控的模型
    from flashinfer import gelu_and_mul

    return gelu_and_mul(x, out=out)


__all__ = ["silu_and_mul", "gelu_and_mul"]
