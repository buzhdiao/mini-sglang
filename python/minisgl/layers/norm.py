# Tuple 用于多返回值注解
from typing import Tuple

import torch

from .base import BaseOP


class RMSNorm(BaseOP):
    # RMSNorm 层：x / sqrt(mean(x^2)+eps) * weight，
    # 相比 LayerNorm 免去均值中心化，是 LLaMA/Qwen 等模型的标准归一化
    def __init__(self, size: int, eps: float) -> None:
        # 延迟导入 flashinfer 的 GPU rmsnorm 核
        from flashinfer import rmsnorm

        # 数值稳定用的 epsilon
        self.eps = eps
        # 可学习的缩放权重
        self.weight = torch.empty(size)
        # 保存核函数引用，避免每次 forward 重复导入
        self.rmsnorm = rmsnorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 标准前向：返回归一化后的新张量
        return self.rmsnorm(x, self.weight, self.eps)

    def forward_inplace(self, x: torch.Tensor) -> None:
        # 原地前向：结果直接写回 x，省一次显存分配（如 Q/K 归一化场景）
        self.rmsnorm(x, self.weight, self.eps, out=x)


class RMSNormFused(BaseOP):
    # 融合残差的 RMSNorm：把 residual 相加与归一化合并成一个 GPU 核，
    # 省去额外的一次读写往返（Transformer 层间残差连接的标准优化）
    def __init__(self, size: int, eps: float) -> None:
        # 延迟导入两个核：普通 rmsnorm 与 fused_add_rmsnorm
        from flashinfer import fused_add_rmsnorm, rmsnorm

        self.eps = eps
        self.weight = torch.empty(size)
        self.rmsnorm = rmsnorm
        self.fused_add_rmsnorm = fused_add_rmsnorm

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 第一层（或无残差）：只做归一化，x 本身作为新的残差返回
            return self.rmsnorm(x, self.weight, self.eps), x
        # 融合路径：residual += x（原地），再对 residual 归一化写回 x，
        # 返回 (归一化输出, 累积残差)，两个结果都在原缓冲上更新
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual
