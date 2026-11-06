# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
# 全局上下文：forward 从中取 batch 的 input_ids
from minisgl.core import get_global_ctx
# 基础算子与模型所需层
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding
# NVTX 性能打点
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
# Mistral 与 Llama 结构一致，复用通用注意力与 MLP
from .utils import GatedMLP as MistralMLP
from .utils import RopeAttn as MistralAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class MistralDecoderLayer(BaseOP):
    # Mistral 解码器单层：与 Llama 相同的残差结构
    def __init__(self, config: ModelConfig, layer_id: int):
        # 自注意力（RoPE + GQA）
        self.self_attn = MistralAttn(config, layer_id)
        # 门控 FFN（SwiGLU）
        self.mlp = MistralMLP(config)
        # 注意力前归一化（融合残差）
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        # MLP 前归一化
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # 层号供 NVTX 打点用
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 残差累加 + 归一化，得到注意力输入
        x, residual = self.input_layernorm.forward(x, residual)
        # 自注意力
        x = self.self_attn.forward(x)
        # 输出并入残差并归一化，得到 MLP 输入
        x, residual = self.post_attention_layernorm.forward(x, residual)
        # 门控 FFN
        x = self.mlp.forward(x)
        # 返回 (输出, 累积残差)
        return x, residual


class MistralModel(BaseOP):
    # Mistral 主体：词嵌入 + N 层解码层 + 最终归一化
    def __init__(self, config: ModelConfig):
        # 词表并行嵌入
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # 聚合各解码层
        self.layers = OPList(
            [MistralDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        # 最终 RMSNorm
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # token id -> 词向量
        x = self.embed_tokens.forward(input_ids)
        # 残差流初始为空
        residual: torch.Tensor | None = None
        # 逐层前向
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        # 残差并入并归一化
        return self.norm.forward(x, residual)[0]


class MistralForCausalLM(BaseLLMModel):
    # 因果 LM 顶层：主体 + LM 输出头
    def __init__(self, config: ModelConfig):
        # 模型主体
        self.model = MistralModel(config)
        # LM 头（支持与嵌入绑定权重）
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # 取当前 batch 的输入 token
        ids = get_global_ctx().batch.input_ids
        # 主体前向得到隐藏态
        output = self.model.forward(ids)
        # 投影到词表得到 logits
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["MistralForCausalLM"]