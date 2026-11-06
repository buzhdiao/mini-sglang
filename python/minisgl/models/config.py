# 延迟类型注解：注解在运行时不求值
from __future__ import annotations
# dataclass 生成不可变配置结构
from dataclasses import dataclass
from typing import Any, Dict
# HF transformers 的预训练配置类型
from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    # RoPE 相关配置（不可变）
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    # 从 HF config 提炼出的模型结构配置（不可变），供建模型/KV cache 使用
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]

    @property
    def is_moe(self) -> bool:
        # 依据 model_type 判断是否为 MoE 模型（如 qwen3_moe）
        return "moe" in self.model_type

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        # 从 HF 预训练配置解析出本引擎的模型配置
        if hasattr(config, "text_config") and config.text_config is not None:
            # 多模态模型（如 Mistral3）的文本结构藏在 text_config 子配置里
            top = config
            config = config.text_config
            # 顶层才有的架构/RoPE 字段补到子配置上，统一后续读取路径
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        # GQA 的 KV 头数：缺失时（MHA）等于 Q 头数
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        # 单头维度：部分模型直接给 head_dim，否则由隐藏维/Q头数推得
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        # 是否绑定输入嵌入与输出头
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        # 模型类型名（llama/qwen2/qwen3_moe...），缺省 llama
        model_type = getattr(config, "model_type", "llama")
        # MoE 专家数：兼容两种常见字段名；非 MoE 为 0
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        # 每 token 激活专家数
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        # MoE 单专家中间维
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        # 是否对 top-k 路由权重归一化
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        # HF 架构类名，用于查模型注册表
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        # RoPE 缩放配置
        rope_scaling = getattr(config, "rope_scaling", None)
        # RoPE 基数：Llama/Qwen 是直接属性；Mistral 藏在 rope_scaling 字典里
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        # 组装不可变的模型配置对象
        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=head_dim,
                rotary_dim=head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
        )