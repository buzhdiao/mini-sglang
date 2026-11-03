from typing import Tuple  # 类型注解：Tuple 用于标注返回多个张量的元组

import torch  # PyTorch 主库：张量类型与 weight 存储

from .base import BaseOP  # 导入算子基类，获得自定义 state_dict/load_state_dict 协议


class RMSNorm(BaseOP):
    """RMSNorm 层：Root Mean Square 归一化，y = x / rms(x) * weight。

    相比 LayerNorm 免去减均值（只做缩放），计算量更小且效果相近，
    是 LLaMA/Qwen 等现代 LLM 的标配。权重通过 flashinfer 的 CUDA kernel 融合计算。
    """

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import rmsnorm  # 延迟导入 flashinfer 的 RMSNorm CUDA kernel，加快模块加载

        self.eps = eps  # 数值稳定项：防止除零，加在均方根的分母中（通常 1e-5 / 1e-6）
        self.weight = torch.empty(size)  # 可学习的缩放权重，形状 [size]（每个隐藏维度一个），初始为空占位、由 checkpoint 加载填充
        self.rmsnorm = rmsnorm  # 把 kernel 函数存为实例属性，避免 forward 每次调用时查找全局名

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rmsnorm(x, self.weight, self.eps)  # 调用 CUDA kernel：输入 x 形状 [tokens, size]，返回归一化后的新张量

    def forward_inplace(self, x: torch.Tensor) -> None:
        self.rmsnorm(x, self.weight, self.eps, out=x)  # 原地版本：结果写回 x 自身，省一次显存分配（attention 层对 q/k 归一化时使用）


class RMSNormFused(BaseOP):
    """残差融合版 RMSNorm：把 residual add 与 norm 融合为一个 kernel，减少一次显存读写。

    forward 语义随 residual 是否为 None 分两种：
    - residual=None（每层第一次调用）：只做 norm，x 原样作为新的 residual 返回；
    - residual 非 None：执行 x += residual 后再 norm，x/residual 均被原地更新。
    """

    def __init__(self, size: int, eps: float) -> None:
        from flashinfer import fused_add_rmsnorm, rmsnorm  # 延迟导入两个 kernel：普通 rmsnorm 与 加法融合 rmsnorm

        self.eps = eps  # 数值稳定项，与 RMSNorm 相同
        self.weight = torch.empty(size)  # 可学习缩放权重，形状 [size]
        self.rmsnorm = rmsnorm  # 保留普通 kernel：处理 residual 为 None 的首层路径
        self.fused_add_rmsnorm = fused_add_rmsnorm  # 融合 kernel：一步完成 residual 相加与归一化

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None  # x 为本层输入；residual 为跨层残差流（首次为 None）
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回 (norm 后的输出, 更新后的 residual)，供下一层继续使用
        if residual is None:  # 首个 norm 层：残差流尚未建立
            return self.rmsnorm(x, self.weight, self.eps), x  # 输出 = norm(x)，同时原 x 充当初始 residual（拷贝引用，kernel 返回新张量）
        self.fused_add_rmsnorm(x, residual, self.weight, self.eps)  # 融合 kernel：x += residual，再对 x 做 RMSNorm（x、residual 均原地更新）
        return x, residual  # 返回归一化输出与更新后的残差流，与 Transformer 残差结构对应
