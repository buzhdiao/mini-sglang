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
# 注意力与 Llama/Qwen3 相同；FFN 换成 MoE 版本（MoEMLP）
from .utils import MoEMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    # Qwen3-MoE 解码器单层：注意力同 Qwen3，FFN 换成稀疏专家
    def __init__(self, config: ModelConfig, layer_id: int):
        # 自注意力：Qwen3-MoE 同样使用 QK-Norm
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        # MoE FFN：路由 top-k 专家，替代稠密 MLP
        self.mlp = Qwen3MLP(config)
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
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 残差累加 + 归一化，得到注意力输入
        x, residual = self.input_layernorm.forward(x, residual)
        # 自注意力
        x = self.self_attn.forward(x)
        # 输出并入残差并归一化，得到 MoE 输入
        x, residual = self.post_attention_layernorm.forward(x, residual)
        # MoE 前向：mlp.forward 内部先算路由 logits，再走融合专家核
        x = self.mlp.forward(x)
        # 返回 (输出, 累积残差)
        return x, residual


class Qwen3Model(BaseOP):
    # Qwen3-MoE 主体：词嵌入 + N 层解码层 + 最终归一化
    def __init__(self, config: ModelConfig):
        # 词表并行嵌入
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # 聚合各解码层（每层内含 MoE）
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
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


class Qwen3MoeForCausalLM(BaseLLMModel):
    # 因果 LM 顶层：主体 + LM 输出头
    def __init__(self, config: ModelConfig):
        # 模型主体
        self.model = Qwen3Model(config)
        # LM 头（支持与嵌入绑定权重）
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
        )
        super().__init__()

    def forward(self) -> torch.Tensor:
        # 取当前 batch 输入并做主体前向
        output = self.model.forward(get_global_ctx().batch.input_ids)
        # 投影到词表得到 logits
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3MoeForCausalLM"]
