# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

from typing import Dict

import torch
# F.linear 用于 LM head 的矩阵乘
import torch.nn.functional as F
# 全局上下文：提供 batch 信息
from minisgl.core import get_global_ctx
# 分布式通信器与 TP 信息：词表并行需要 all_reduce/all_gather
from minisgl.distributed import DistributedCommunicator, get_tp_info
# div_ceil 向上取整除法；nvtx_annotate 用于性能分析打点
from minisgl.utils import div_ceil, nvtx_annotate

from .base import BaseOP


class VocabParallelEmbedding(BaseOP):
    # 词表并行嵌入：把词表按行切成 TP 份，每个 rank 只存自己负责的词向量，
    # 超出范围的 token 查表得 0，最后 all_reduce 求和得到完整嵌入
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 获取 TP 并行信息
        tp_info = get_tp_info()
        # 本 rank 编号
        tp_rank = tp_info.rank
        # TP 规模
        self.tp_size = tp_info.size
        # 全局词表大小
        self.num_embeddings = num_embeddings
        # 每 rank 分到的词表行数（向上取整，保证覆盖全部词表）
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)
        # 本 rank 负责的词表起始下标
        start_idx = self.num_embeddings_tp * tp_rank
        # 结束下标（末位 rank 可能不满）
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        # 本 rank 的词表区间 (起始, 长度)
        self.vocab_range = (start_idx, finish_idx - start_idx)
        # 本 rank 的嵌入权重分片
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        # TP 通信器
        self._comm = DistributedCommunicator()

    @nvtx_annotate("Embedding")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向：按 token id 查本 rank 的词向量分片
        from minisgl.kernel import indexing

        # 用 GPU 核按索引取行；vocab_range 供核把越界 token（不属于本 rank）置 0
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        # 多卡时各 rank 只有部分词向量，all_reduce 求和还原完整嵌入
        return self._comm.all_reduce(y) if self.tp_size > 1 else y


class ParallelLMHead(VocabParallelEmbedding):
    # LM 输出头：复用词表并行的权重分片，但前向是 GEMM（而非查表），
    # 且需要 all_gather 把各 rank 的 logits 拼回完整词表
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
        tie_word_embeddings: bool = False,
        tied_embedding: VocabParallelEmbedding | None = None,
    ):
        # 初始化与嵌入层相同的词表分片权重
        super().__init__(num_embeddings, embedding_dim)
        # 可选的输出偏置（按分片大小分配）
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None
        # 词绑定（tied embedding）：直接复用输入嵌入的权重
        self.tied_embedding = tied_embedding
        # tie 开关与传入的嵌入对象必须一致
        assert (tied_embedding is not None) == tie_word_embeddings

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        # 权重加载：绑定嵌入时跳过 lm_head 自己的权重
        if not self.tied_embedding:
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        else:
            # pop the lm_head.weights and lm_head.bias if they exist
            # 绑定模式下权重与嵌入共享，检查点里的 lm_head 参数是冗余的，直接丢弃
            possible_weight = f"{prefix}.weight"
            possible_bias = f"{prefix}.bias"
            if possible_weight in state_dict:
                state_dict.pop(possible_weight)
            if possible_bias in state_dict:
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",
        result: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        # 导出：绑定模式下不导出 lm_head 权重（与嵌入共享，避免重复）
        if not self.tied_embedding:
            return super().state_dict(prefix=prefix, result=result)
        return {} if result is None else result

    @nvtx_annotate("LMHead")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 取全局上下文与 batch
        ctx = get_global_ctx()
        batch = ctx.batch
        # batch 内 token 数
        bs = batch.size
        if batch.is_prefill:
            # 预填充阶段：序列只有最后一个 token 需要 logits，
            # 用元数据取每条序列最后 token 的下标，只对它们做输出投影
            indices = batch.attn_metadata.get_last_indices(bs)
            x = x[indices].contiguous()
            del indices

        # 绑定嵌入时用嵌入的权重分片，否则用自己的
        module = self.tied_embedding or self
        # 与本 rank 词表分片做线性投影：(token, hidden) -> (token, 词表分片)
        logits = F.linear(x, module.weight, self.bias)
        if self.tp_size == 1:
            # 单卡：词表完整，直接返回
            return logits
        # 记下分片 logits 形状
        input_shape = logits.shape
        # all_gather 把各 rank 的词表分片拼成完整词表维
        output_tensor = self._comm.all_gather(logits)

        if bs == 1:
            # 单 token：all_gather 后是一维拼接，直接裁剪到真实词表大小
            return output_tensor.view(1, -1)[:, : self.num_embeddings]

        # 多 token：gather 结果按 rank 维堆叠，需重排成 (token, tp*分片) 再裁剪
        output_tensor = output_tensor.view((self.tp_size,) + input_shape)
        # 把 rank 维挪到词表维之前
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()
        # 摊平成 (token, tp_size*词表分片)
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))
        # 裁掉向上取整产生的填充行，得到完整词表的 logits
        return output_tensor[:, : self.num_embeddings]
