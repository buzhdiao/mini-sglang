# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# functools.cache 用于缓存 RoPE 实例，避免重复构造查表
import functools
# math 用于 llama3/yarn 缩放公式的对数与圆周率
import math
from typing import Any, Callable, Dict, Tuple

import torch

from .base import StateLessOP


class RotaryEmbedding(StateLessOP):
    # 旋转位置编码（RoPE）：把 Q/K 的每个二维子空间按 token 位置旋转，
    # 使注意力分数只依赖相对位置。cos/sin 预计算成查表，前向时按位置索引应用
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # 单头维度
        self.head_size = head_size
        # 当前实现要求全头维度都参与旋转（不部分旋转）
        assert rotary_dim == head_size
        # 每个频率分量对应的逆频率：base^(-2i/d)，低维转得快、高维转得慢
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        if post_process is not None:
            # 允许外挂频率修正（llama3/yarn 等长度外推方案）
            inv_freq = post_process(inv_freq)
        # 生成 0..max_position-1 的绝对位置序列
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        # 外积得到 (位置, 频率) 相位矩阵
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        # 相位的余弦
        cos = freqs.cos()
        # 相位的正弦
        sin = freqs.sin()
        # buffer, so don't load/save
        # cos/sin 拼成查表缓存；这是推导出的常量表而非可学习参数，因此不参与保存/加载
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)
        # flashinfer 的 RoPE 核只支持这些头维
        assert self.head_size in [64, 128, 256, 512]

        # 延迟导入 flashinfer 的原地 RoPE 应用核
        from flashinfer import apply_rope_with_cos_sin_cache_inplace

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 按每个 token 的位置从查表取 cos/sin，原地旋转 Q/K
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self._cos_sin_cache,
        )
        # 原地更新后直接返回同一对张量
        return query, key


def _get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Dict[str, Any] | None = None,
) -> RotaryEmbedding:
    # RoPE 工厂：按模型配置选择标准实现或某种长度外推缩放
    if rope_scaling is None:
        # 无缩放：标准 RoPE
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)
    # need to test some cases:
    # 按 rope_type 分派不同的频率修正方案
    match rope_scaling["rope_type"]:
        case "default":
            # 声明为 default 等价于无缩放
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)

        case "llama3":
            # llama3 频率插值：低频分量保持不动，高频分量按 factor 压缩，
            # 中间频率平滑过渡，从而在扩展上下文时保留高频分辨率
            scaling_factor: float = rope_scaling["factor"]
            low_freq_factor: float = rope_scaling["low_freq_factor"]
            high_freq_factor: float = rope_scaling["high_freq_factor"]
            original_max_position: int = rope_scaling["original_max_position_embeddings"]

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # no smooth if low_freq_factor == high_freq_factor
                # 由逆频率反推波长
                wave_len = 2 * math.pi / inv_freq
                if low_freq_factor == high_freq_factor:
                    # 无平滑区：波长小于阈值的频率不变，否则整体缩放
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,
                        inv_freq,
                        inv_freq / scaling_factor,
                    )

                # 平滑区间宽度
                delta = high_freq_factor - low_freq_factor
                # 计算平滑系数：0 表示不缩放，1 表示完全缩放
                smooth = (original_max_position / wave_len - low_freq_factor) / delta
                smooth = torch.clamp(smooth, 0, 1)
                # 在 1 与 1/factor 之间插值出实际缩放因子
                factor = (1 - smooth) / scaling_factor + smooth
                return factor * inv_freq

            # 构造带频率修正的 RoPE
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

        case "yarn":
            # YaRN：按"旋转圈数"划界，把低频分量的波长插值拉长 factor 倍，
            # 高频分量保持不变，中间用斜坡函数线性过渡
            factor: float = rope_scaling["factor"]
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]

            def _find_correction_dim(num_rotations: float) -> float:
                # 求"在原上下文内恰好转 num_rotations 圈"的频率维下标
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

            # 高频边界：转得快于 beta_fast 圈的维不动
            low = max(math.floor(_find_correction_dim(beta_fast)), 0)
            # 低频边界：转得慢于 beta_slow 圈的维完全插值
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                # 各频率维在 [low, high] 上的 0→1 斜坡
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),
                    0, 1,
                )
                # 低频侧缩放 1/factor，高频侧保持原值，中间线性混合
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)

    # 未知的 rope_type 直接报错
    raise ValueError(f"Unsupported {rope_scaling = }")


# RoPE 查表所在设备的全局记录
_ROPE_DEVICE: torch.device | None = None


def set_rope_device(device: torch.device):
    # 设置 RoPE 缓存的设备（模型权重在 meta 设备上时，RoPE 不能跟着用 meta）
    global _ROPE_DEVICE
    _ROPE_DEVICE = device


@functools.cache
def get_rope(
    head_dim: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,
) -> RotaryEmbedding:
    # 带缓存的 RoPE 构造入口：参数相同就复用同一实例
    rope_map = dict(rope_scaling) if rope_scaling is not None else None
    # 用一个 meta 设备张量探测当前默认设备
    t = torch.tensor([])
    if t.device == torch.device("meta"):
        # we cannot use meta device for rope
        # RoPE 查表需要真实数值，不能放 meta 设备；必须先用 set_rope_device 指定目标设备
        if _ROPE_DEVICE is None:
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."
            )
        # 在指定设备上下文里构造 RoPE
        with torch.device(_ROPE_DEVICE):
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    # 默认设备正常构造
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]