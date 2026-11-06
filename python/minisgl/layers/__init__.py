# 激活函数（SiLU/GELU 与门控相乘的融合算子）
from .activation import gelu_and_mul, silu_and_mul
# 注意力层：QKV 切分 + RoPE + 后端注意力计算
from .attention import AttentionLayer
# 算子基类：自定义轻量 state_dict 机制（替代 torch.nn.Module）
from .base import BaseOP, OPList, StateLessOP
# 词嵌入与 LM 输出头（支持词表并行）
from .embedding import ParallelLMHead, VocabParallelEmbedding
# 各类线性层：列并行（合并 QKV/OProj 变体）、行并行、复制式
from .linear import (
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
)
# MoE 层：路由 + 专家计算的封装
from .moe import MoELayer
# RMSNorm（含 fused 变体）
from .norm import RMSNorm, RMSNormFused
# 旋转位置编码（RoPE）构造与设备绑定
from .rotary import get_rope, set_rope_device

# 对外暴露的公共 API
__all__ = [
    "silu_and_mul",
    "gelu_and_mul",
    "AttentionLayer",
    "BaseOP",
    "StateLessOP",
    "OPList",
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "LinearColParallelMerged",
    "LinearRowParallel",
    "LinearOProj",
    "LinearQKVMerged",
    "RMSNorm",
    "RMSNormFused",
    "get_rope",
    "set_rope_device",
    "LinearReplicated",
    "MoELayer",
]
