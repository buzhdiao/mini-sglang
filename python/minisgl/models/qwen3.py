from __future__ import annotations  # 启用延迟注解求值，允许类型注解使用前向引用

from typing import TYPE_CHECKING, Tuple  # TYPE_CHECKING：仅静态检查时导入重模块；Tuple：forward 返回值类型

import torch  # PyTorch 主库：本文件仅用于 forward 参数/返回值的类型注解
from minisgl.core import get_global_ctx  # 全局上下文：拿当前正在执行的 batch（含 input_ids 等）
from minisgl.layers import BaseOP, OPList, ParallelLMHead, RMSNormFused, VocabParallelEmbedding  # 层组件：算子基类/算子列表/并行输出头/融合 RMSNorm/词并行 embedding
from minisgl.utils import nvtx_annotate  # NVTX 性能标注装饰器：在 nsight profiler 中按层显示耗时区间

from .base import BaseLLMModel  # LLM 抽象基类
from .utils import GatedMLP as Qwen3MLP  # 门控 MLP（SwiGLU 结构），别名 Qwen3MLP：Qwen3 的稠密 FFN
from .utils import RopeAttn as Qwen3Attn  # RoPE 注意力层，别名 Qwen3Attn：Qwen3 的自注意力

if TYPE_CHECKING:  # 仅类型检查时导入，避免运行时循环依赖（config 导入本包）
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    """Qwen3 单层解码器 = RMSNorm → 自注意力 → RMSNorm → 门控 MLP（Pre-Norm 残差结构）。

    与 Llama/Qwen2 的差别：Qwen3 注意力带 Q/K 归一化（has_qk_norm=True），
    QKV 投影无 bias（Qwen2 则相反：有 bias、无 QK Norm）。
    """

    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)  # Qwen3 注意力：开启 Q/K RMSNorm（稳定大尺度 QK 内积）、无 QKV 偏置
        self.mlp = Qwen3MLP(config)  # FFN 子层：gate/up 合并投影 + SiLU 门控 + down 投影
        self.input_layernorm = RMSNormFused(  # 注意力前归一化：融合"归一化 + 残差相加"两个操作
            size=config.hidden_size,  # 归一化的特征维度（每个 token 向量宽度）
            eps=config.rms_norm_eps,  # 数值稳定小量，防除零
        )
        self.post_attention_layernorm = RMSNormFused(  # MLP 前归一化：同样融合残差相加
            size=config.hidden_size,  # 同上
            eps=config.rms_norm_eps,  # 同上
        )

        self._layer_id = layer_id  # 保存层编号，供 nvtx_annotate 装饰器填充性能标注名

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")  # 性能标注：生成如 "Layer_5" 的 NVTX 区间名，方便 profiler 定位每层耗时
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None  # 输入 hidden states [num_tokens, hidden]；residual 首层为 None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)  # 归一化 x 并与 residual 相加；首层 residual=None 则只归一化
        x = self.self_attn.forward(x)  # 自注意力：[num_tokens, hidden] → [num_tokens, hidden]
        x, residual = self.post_attention_layernorm.forward(x, residual)  # MLP 前归一化：residual 保存注意力输出，x 为归一化结果
        x = self.mlp.forward(x)  # FFN：[num_tokens, hidden] → [num_tokens, hidden]
        return x, residual  # 返回 (MLP 输出, 残差流=注意力输出)；残差相加延迟到下一层的 norm 中完成


class Qwen3Model(BaseOP):
    """Qwen3 主体：embedding + N 层解码器 + 最终 RMSNorm（不含 lm_head）。"""

    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(  # 词嵌入：按词表维度做 TP 切分，每个 rank 持有一部分词表行
            num_embeddings=config.vocab_size,  # 词表大小（embedding 矩阵行数）
            embedding_dim=config.hidden_size,  # 嵌入维度（列数）
        )
        self.layers = OPList(  # 算子列表容器：统一管理 N 个解码器层
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]  # 逐层构建，layer_id 用于 KV Cache 定位与性能标注
        )
        self.norm = RMSNormFused(  # 最终归一化：融合最后的残差相加 + RMSNorm
            size=config.hidden_size,  # 特征维度
            eps=config.rms_norm_eps,  # eps
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """前向：token ids → 词嵌入 → N 层解码 → 最终归一化，输出 hidden states。"""
        x = self.embed_tokens.forward(input_ids)  # 查词表：[num_tokens] 整型 id → [num_tokens, hidden_size]
        residual: torch.Tensor | None = None  # 残差流初始为 None：第一层的 norm 只归一化、不做相加
        for layer in self.layers.op_list:  # 顺序执行每层解码器
            x, residual = layer.forward(x, residual)  # 传入 (hidden, residual)，取回新的 (hidden, residual)
        return self.norm.forward(x, residual)[0]  # 最终：residual+x 相加后归一化；[0] 取归一化结果（返回元组的第一个元素）


class Qwen3ForCausalLM(BaseLLMModel):
    """Qwen3 因果语言模型完整封装：主体 + 输出头，实现 BaseLLMModel.forward 接口。"""

    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)  # transformer 主体（embedding+层+norm）
        self.lm_head = ParallelLMHead(  # 输出投影头：hidden → vocab logits，支持 TP 按词表切分
            num_embeddings=config.vocab_size,  # 词表大小（输出矩阵行数）
            embedding_dim=config.hidden_size,  # 输入维度
            tie_word_embeddings=config.tie_word_embeddings,  # Qwen3 小尺寸模型（0.6B/1.7B）通常绑定词嵌入
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,  # 绑定时直接引用 embed_tokens（同一块权重两用）
        )
        super().__init__()  # 调用 BaseOP/BaseLLMModel 初始化（收集子模块等；注意须在子模块赋值之后）

    def forward(self) -> torch.Tensor:
        """前向：从全局上下文取当前 batch 的 input_ids，输出 logits。"""
        output = self.model.forward(get_global_ctx().batch.input_ids)  # 全局上下文拿 input_ids，跑主体得 hidden states
        logits = self.lm_head.forward(output)  # hidden states → 词表 logits：[num_tokens, vocab_size]
        return logits  # 返回 logits，由采样器决定下一个 token


__all__ = ["Qwen3ForCausalLM"]  # 模块对外只暴露完整模型类（register.py 按此类名实例化）
