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
# 复用通用的门控 MLP 与注意力实现（Llama 与 Mistral 结构相同）
from .utils import GatedMLP as LlamaMLP
from .utils import RopeAttn as LlamaAttn

if TYPE_CHECKING:
    from .config import ModelConfig


class LlamaDecoderLayer(BaseOP):
    # Llama 解码器单层：残差式"归一化->注意力->归一化->MLP"
    def __init__(self, config: ModelConfig, layer_id: int):
        # 自注意力（含 QKV/OProj 与 RoPE）
        self.self_attn = LlamaAttn(config, layer_id)
        # 门控 FFN（SwiGLU：gate_up 投影 + SiLU 门控 + down 投影）
        self.mlp = LlamaMLP(config)
        # 注意力前归一化（融合残差的 RMSNorm）
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        # MLP 前归一化
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # 层号供 NVTX 打点用（下划线开头不进 state_dict）
        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 融合步骤：residual += x，再对和做 RMSNorm 得到注意力输入
        x, residual = self.input_layernorm.forward(x, residual)
        # 自注意力（内部读写该层 KV cache）
        x = self.self_attn.forward(x)
        # 注意力输出先并入残差流，再归一化出 MLP 输入
        x, residual = self.post_attention_layernorm.forward(x, residual)
        # 门控 FFN
        x = self.mlp.forward(x)
        # 返回 (待并入残差的输出, 累积残差)，下一层继续
        return x, residual


class LlamaModel(BaseOP):
    # Llama 主体：词嵌入 + N 层解码层 + 最终归一化
    def __init__(self, config: ModelConfig):
        # 词表并行嵌入
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # 聚合各解码层（参数名 layers.0/1/2...）
        self.layers = OPList(
            [LlamaDecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        # 最终 RMSNorm（输出前）
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # token id -> 词向量
        x = self.embed_tokens.forward(input_ids)
        # 残差流初始为空（第一层从嵌入值起算）
        residual: torch.Tensor | None = None
        # 逐层前向，维护 (隐藏态, 残差) 二元组
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        # 最后把残差并入并归一化，取归一化结果返回
        return self.norm.forward(x, residual)[0]


class LlamaForCausalLM(BaseLLMModel):
    # 因果 LM 顶层：主体 + LM 输出头
    def __init__(self, config: ModelConfig):
        # 模型主体
        self.model = LlamaModel(config)
        # LM 头（与嵌入共享权重时直接复用嵌入对象）
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # 从全局上下文取当前 batch 的输入 token
        output = self.model.forward(get_global_ctx().batch.input_ids)
        # 投影到词表得到 logits
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["LlamaForCausalLM"]
