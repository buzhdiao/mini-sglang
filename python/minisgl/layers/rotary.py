from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如 torch.device | None）

import functools  # 提供 functools.cache：按参数缓存 get_rope 工厂结果，避免重复构建 RoPE 表
import math  # 数学库：RoPE 缩放（llama3/yarn）中的对数与圆周率计算
from typing import Any, Callable, Dict, Tuple  # 类型注解工具

import torch  # PyTorch 主库：构建 cos/sin 缓存表与张量运算

from .base import StateLessOP  # 无状态算子基类：RoPE 无可学习权重（cos/sin 是派生缓存，不进 checkpoint）


class RotaryEmbedding(StateLessOP):
    """旋转位置编码（RoPE）：把 token 位置信息以旋转矩阵形式注入 q/k 的内积中。

    预先计算好 [max_position, rotary_dim] 的 cos/sin 缓存表，前向时按每个 token
    的 position 查表，由 flashinfer kernel 原地旋转 q/k。
    """

    def __init__(
        self,
        head_size: int,  # 每个注意力头的维度
        rotary_dim: int,  # 参与旋转的维度（可小于 head_size，其余维度不旋转）
        max_position_embeddings: int,  # 支持的最大位置数（缓存表长度）
        base: float,  # 频率基数 theta（通常 10000，不同频率分量 = theta^(-2i/d)）
        post_process: None | Callable[[torch.Tensor], torch.Tensor] = None,  # 可选的频率后处理（llama3/yarn 外推用）
    ) -> None:
        super().__init__()  # 调用基类构造
        self.head_size = head_size  # 保存头维度，forward 时传给 kernel
        assert rotary_dim == head_size  # 当前实现仅支持全维旋转（partial rotary 未启用）
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))  # 每对分量的基础频率：theta^(-2i/d)，形状 [rotary_dim/2]，低维高频、高维低频
        if post_process is not None:  # 配置了频率缩放（llama3/yarn）
            inv_freq = post_process(inv_freq)  # 对基础频率做外推修正后再建表
        t = torch.arange(max_position_embeddings, dtype=torch.float)  # 位置序列 [0, 1, ..., max_pos-1]，形状 [max_pos]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)  # 外积得每个 (位置, 频率) 的相位角，形状 [max_pos, rotary_dim/2]
        cos = freqs.cos()  # 余弦分量，旋转矩阵的实部
        sin = freqs.sin()  # 正弦分量，旋转矩阵的虚部
        # buffer, so don't load/save
        self._cos_sin_cache = torch.cat((cos, sin), dim=-1)  # 拼成 [max_pos, rotary_dim] 缓存表（前半 cos 后半 sin）；下划线开头使其不被 state_dict 收集——它可由配置派生，无需存 checkpoint
        assert self.head_size in [64, 128, 256, 512]  # flashinfer kernel 仅对这些头维度做了特化

        from flashinfer import apply_rope_with_cos_sin_cache_inplace  # 延迟导入 flashinfer 的原地 RoPE kernel

        self.apply_rope_with_cos_sin_cache_inplace = apply_rope_with_cos_sin_cache_inplace  # 存为实例属性，避免 forward 每次全局查找

    def forward(
        self,
        positions: torch.Tensor,  # 每个 token 的绝对位置 id，形状 [tokens]
        query: torch.Tensor,  # q 张量，形状 [tokens, num_heads * head_size]，将被原地旋转
        key: torch.Tensor,  # k 张量，形状 [tokens, num_kv_heads * head_size]，将被原地旋转
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回旋转后的 (q, k)（与输入同对象）
        self.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,  # 按位置查缓存表
            query=query,  # 原地旋转 q
            key=key,  # 原地旋转 k
            head_size=self.head_size,  # 头维度
            cos_sin_cache=self._cos_sin_cache,  # 预计算的 [max_pos, rotary_dim] 余弦/正弦表
        )
        return query, key  # kernel 原地修改，直接返回原张量


def _get_rope(
    head_dim: int,  # 头维度
    rotary_dim: int,  # 旋转维度
    max_position: int,  # 最大位置（含缩放后扩展到的长度）
    base: float,  # 频率基数
    rope_scaling: Dict[str, Any] | None = None,  # 缩放配置字典（rope_type 为键之一）
) -> RotaryEmbedding:
    """RoPE 工厂：无缩放时直接构建；按 rope_type 分派 llama3/yarn 的频率修正策略。"""
    if rope_scaling is None:  # 未配置缩放
        return RotaryEmbedding(head_dim, rotary_dim, max_position, base)  # 标准 RoPE
    # need to test some cases:
    match rope_scaling["rope_type"]:  # 按缩放类型分派（Python 3.10 结构化匹配）
        case "default":  # 声明为 default：等同无缩放
            return RotaryEmbedding(head_dim, rotary_dim, max_position, base)  # 标准 RoPE

        case "llama3":  # Llama 3 的频率缩放：低频分量压缩、高频保留
            scaling_factor: float = rope_scaling["factor"]  # 上下文扩展倍数
            low_freq_factor: float = rope_scaling["low_freq_factor"]  # 低频分量的缩放边界因子
            high_freq_factor: float = rope_scaling["high_freq_factor"]  # 高频分量的缩放边界因子
            original_max_position: int = rope_scaling["original_max_position_embeddings"]  # 原始（预训练时）最大位置

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                """对基础频率 inv_freq 应用 llama3 分段/平滑缩放，返回修正后的频率。"""
                # no smooth if low_freq_factor == high_freq_factor
                wave_len = 2 * math.pi / inv_freq  # 每个频率分量的"波长"（旋转一整圈需要的 token 数）
                if low_freq_factor == high_freq_factor:  # 边界重合：退化为硬分段（无平滑过渡）
                    return torch.where(
                        wave_len < original_max_position / high_freq_factor,  # 波长小于阈值（高频分量）
                        inv_freq,  # 高频保持不变
                        inv_freq / scaling_factor,  # 低频压缩 factor 倍（等价于把位置放大 factor 倍）
                    )

                delta = high_freq_factor - low_freq_factor  # 平滑区间的宽度
                smooth = (original_max_position / wave_len - low_freq_factor) / delta  # 按"频率因子"线性插值出平滑系数
                smooth = torch.clamp(smooth, 0, 1)  # 截断到 [0,1]：低于低频界全缩放、高于高频界不缩放
                factor = (1 - smooth) / scaling_factor + smooth  # 逐分量混合缩放因子：smooth=0 全压缩，smooth=1 不变
                return factor * inv_freq  # 应用缩放

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)  # 用修正后的频率构建 RoPE

        case "yarn":  # YaRN（Yet another RoPE extensioN）：按频率分带插值，兼顾外推能力与短程精度
            factor: float = rope_scaling["factor"]  # 上下文扩展倍数
            beta_fast: float = rope_scaling.get("beta_fast", 32.0)  # 高频带边界（旋转圈数）
            beta_slow: float = rope_scaling.get("beta_slow", 1.0)  # 低频带边界（旋转圈数）
            orig_max_pos: int = rope_scaling["original_max_position_embeddings"]  # 原始最大位置

            def _find_correction_dim(num_rotations: float) -> float:
                """求"在原上下文内恰好旋转 num_rotations 圈"的频率维度索引（可为小数）。"""
                return rotary_dim * math.log(orig_max_pos / (num_rotations * 2 * math.pi)) / (2 * math.log(base))  # 由 inv_freq_i = base^(-2i/d) 反解维度 i

            low = max(math.floor(_find_correction_dim(beta_fast)), 0)  # 高频带下界维度：低于它的完全插值（频率 /factor）
            high = min(math.ceil(_find_correction_dim(beta_slow)), rotary_dim // 2 - 1)  # 低频带下界维度：高于它的完全不插值（保持原频率）

            def post_process(inv_freq: torch.Tensor) -> torch.Tensor:
                """按维度在 [low, high] 间线性斜坡，把 inv_freq 从原值过渡到插值值。"""
                ramp = torch.clamp(
                    (torch.arange(rotary_dim // 2, dtype=torch.float32) - low) / max(high - low, 1),  # 每维的斜坡位置（0..1）；max 防止 high==low 除零
                    0, 1,  # 截断到 [0,1]
                )
                return (inv_freq / factor) * ramp + inv_freq * (1 - ramp)  # 斜坡加权：低维(高频)取 /factor 插值，高维(低频)取原频率

            return RotaryEmbedding(head_dim, rotary_dim, max_position, base, post_process)  # 用 yarn 修正频率构建 RoPE

    raise ValueError(f"Unsupported {rope_scaling = }")  # 未知的 rope_type：f-string 里 {rope_scaling = } 会带等号打印变量名与值


_ROPE_DEVICE: torch.device | None = None  # 模块级全局：RoPE 缓存表应创建的目标设备（避免 meta 设备）


def set_rope_device(device: torch.device):
    """设置 RoPE 缓存表的创建设备：模型权重常在 meta 设备上初始化，但 RoPE 表需要真实数据。"""
    global _ROPE_DEVICE  # 声明修改模块级全局变量
    _ROPE_DEVICE = device  # 记录目标设备（如 cuda:0），供 get_rope 使用


@functools.cache  # 按参数缓存：同配置多层调用（每层 attention 各调一次）只构建一份 RoPE 实例
def get_rope(
    head_dim: int,  # 头维度
    rotary_dim: int,  # 旋转维度
    max_position: int,  # 最大位置
    base: float,  # 频率基数
    rope_scaling: Tuple[Tuple[str, Any], ...] | None = None,  # 缩放配置（tuple 形式才可哈希，满足缓存要求）
) -> RotaryEmbedding:
    """RoPE 获取入口：处理 meta 设备降级后调用工厂；同参数共享缓存实例。"""
    rope_map = dict(rope_scaling) if rope_scaling is not None else None  # tuple 转回 dict 供工厂内部按键取值
    t = torch.tensor([])  # 构造一个探测张量：读取当前默认设备类型
    if t.device == torch.device("meta"):  # 默认设备是 meta（权重初始化阶段的标志）：meta 上无法算 cos/sin 数值
        # we cannot use meta device for rope
        if _ROPE_DEVICE is None:  # 未预先设置真实设备
            raise RuntimeError(
                "We cannot use meta device for rope. Please call set_rope_device() first."  # 提示必须先调用 set_rope_device
            )
        with torch.device(_ROPE_DEVICE):  # 在指定真实设备上下文中创建（缓存表落到该设备）
            return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)
    return _get_rope(head_dim, rotary_dim, max_position, base, rope_map)  # 默认设备非 meta：正常创建


__all__ = ["get_rope", "RotaryEmbedding", "set_rope_device"]  # 声明模块公开 API
