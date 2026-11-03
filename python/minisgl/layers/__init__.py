from .activation import gelu_and_mul, silu_and_mul  # 门控激活函数（SwiGLU/GeGLU 的后半段：silu/gelu × 门控）
from .attention import AttentionLayer  # 注意力层封装：QKV 拆分 + 可选 q/k norm + RoPE + attention 后端调用
from .base import BaseOP, OPList, StateLessOP  # 算子基类：自定义轻量 state_dict 协议（区别于 nn.Module）
from .embedding import ParallelLMHead, VocabParallelEmbedding  # 词表并行 embedding 与 LM 输出头
from .linear import (  # 各类张量并行线性层（列并行/行并行/复制/QKV 合并/o_proj）
    LinearColParallelMerged,  # 列并行合并层：按输出维切分，一次算多个子层（如 MLP gate+up）
    LinearOProj,  # 注意力输出投影：行并行，本地乘后 all_reduce
    LinearQKVMerged,  # QKV 合并投影：按注意力头切分的列并行
    LinearReplicated,  # 权重复制（不切分）的线性层
    LinearRowParallel,  # 通用行并行层：按输入维切分，合并后 all_reduce
)
from .moe import MoELayer  # 混合专家层：top-k 路由 + 按专家堆叠的 3D 权重
from .norm import RMSNorm, RMSNormFused  # RMSNorm 及其残差加法融合版本
from .rotary import get_rope, set_rope_device  # RoPE 工厂（带缓存）与其设备设置函数

__all__ = [  # 模块公开 API 列表，约束 from ... import * 的导出范围
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
