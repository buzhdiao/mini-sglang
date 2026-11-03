from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如 VocabParallelEmbedding | None）

from typing import Dict  # 类型注解：Dict 用于 state_dict 的键值对标注

import torch  # PyTorch 主库：张量存储与 F.linear 线性计算
import torch.nn.functional as F  # 函数式接口：LM head 用 F.linear 做矩阵乘
from minisgl.core import get_global_ctx  # 获取全局上下文：forward 中通过它拿到当前 batch 元数据
from minisgl.distributed import DistributedCommunicator, get_tp_info  # TP 通信原语与进程组信息（rank/size）
from minisgl.utils import div_ceil, nvtx_annotate  # div_ceil 向上取整除法；nvtx_annotate 给 CUDA profiling 加标记区间

from .base import BaseOP  # 算子基类：提供 state_dict/load_state_dict 协议


class VocabParallelEmbedding(BaseOP):
    """词表并行 embedding：把词表按行（词）切到各 TP rank，每个 rank 只持有词表分片。

    每个 rank 持有 [vocab_tp, dim] 的分片权重；查表时超出本分片范围的 token
    置 0，随后 all_reduce 求和即得到完整 embedding（正确结果只有分片持有者非零）。
    """

    def __init__(
        self,
        num_embeddings: int,  # 完整词表大小
        embedding_dim: int,  # 每个 token 的 embedding 维度（= hidden_size）
    ):
        super().__init__()  # 调用基类构造
        tp_info = get_tp_info()  # 获取张量并行进程组信息
        tp_rank = tp_info.rank  # 当前进程的 TP rank 编号，决定本 rank 持有词表的哪一段
        self.tp_size = tp_info.size  # TP 并行度（rank 总数）
        self.num_embeddings = num_embeddings  # 保存完整词表大小，后续裁剪 padding 用
        self.num_embeddings_tp = div_ceil(num_embeddings, self.tp_size)  # 本分片词表大小 = ceil(词表/TP数)，向上取整保证覆盖全部词
        start_idx = self.num_embeddings_tp * tp_rank  # 本 rank 分片的起始词 id
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)  # 结束词 id：最后一个 rank 可能不足整片，用 min 裁到词表末尾
        self.vocab_range = (start_idx, finish_idx - start_idx)  # 记录 (起始id, 分片长度)，供 kernel 判断越界 token 并置零
        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)  # 本 rank 的词表分片权重，形状 [词表分片, dim]，空占位待加载
        self._comm = DistributedCommunicator()  # TP 通信器：forward 中用于 all_reduce（下划线开头属性不进 state_dict）

    @nvtx_annotate("Embedding")  # NVTX 标注：在 nsight profiling 时间线上标记该函数区间
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：token id [tokens] -> embedding [tokens, dim]（TP 时经 all_reduce 汇总）。"""
        from minisgl.kernel import indexing  # 延迟导入自研 kernel：带词表范围检查的并行查表

        y = indexing(
            weights=self.weight,  # 本 rank 的词表分片
            indices=x,  # 待查的 token id 张量，形状 [tokens]
            vocab_range=self.vocab_range if self.tp_size > 1 else None,  # TP 时传范围让越界 id 输出 0；单卡无需检查
        )

        return self._comm.all_reduce(y) if self.tp_size > 1 else y  # TP：各 rank 的部分和（仅持有者非零）all_reduce 相加得完整 embedding；单卡直接返回


class ParallelLMHead(VocabParallelEmbedding):
    """LM 输出头：与 embedding 同构（权重即 [vocab_tp, dim]），但前向做反向映射 hidden -> logits。

    支持 tie_word_embeddings：直接复用输入 embedding 的权重（共享存储，省一份词表显存），
    且权重收集/加载交给被绑定的 VocabParallelEmbedding 处理，自身跳过。
    """

    def __init__(
        self,
        num_embeddings: int,  # 词表大小
        embedding_dim: int,  # 隐藏维度
        bias: bool = False,  # 是否带输出偏置（多数 LLM 为 False）
        tie_word_embeddings: bool = False,  # 是否与输入 embedding 共享权重
        tied_embedding: VocabParallelEmbedding | None = None,  # 被绑定的输入 embedding 实例（tie 时必传）
    ):
        super().__init__(num_embeddings, embedding_dim)  # 复用父类：按 TP 切词表并分配空权重（tie 时这份权重不会用到）
        self.bias = torch.empty(self.num_embeddings_tp) if bias else None  # 偏置按词表分片同样切分（与 F.linear 的广播语义匹配）
        self.tied_embedding = tied_embedding  # 保存绑定的 embedding；forward 时借用其 weight
        assert (tied_embedding is not None) == tie_word_embeddings  # 一致性校验：tie 标志与传入实例必须同时存在/同时缺失

    def load_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],  # 权重字典
        *,
        prefix: str = "",  # 键名前缀
        _internal: bool = False,  # 内部递归标志
    ) -> None:
        if not self.tied_embedding:  # 未绑定输入 embedding：走常规加载路径
            return super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)  # 父类按 TP 分片加载 weight/bias
        else:  # 已绑定：本层权重来自输入 embedding，无需加载自身
            # pop the lm_head.weights and lm_head.bias if they exist
            possible_weight = f"{prefix}.weight"  # checkpoint 中可能存在的 lm_head 权重键
            possible_bias = f"{prefix}.bias"  # 可能存在的 lm_head 偏置键
            if possible_weight in state_dict:  # checkpoint 带了 lm_head 权重（与 embedding 数值相同）
                state_dict.pop(possible_weight)  # 直接丢弃，避免残留触发顶层"多余键"报错
            if possible_bias in state_dict:  # 同理丢弃偏置
                state_dict.pop(possible_bias)

    def state_dict(
        self,
        *,
        prefix: str = "",  # 键名前缀
        result: Dict[str, torch.Tensor] | None = None,  # 复用的结果容器
    ) -> Dict[str, torch.Tensor]:
        if not self.tied_embedding:  # 未绑定：正常导出自身权重
            return super().state_dict(prefix=prefix, result=result)  # 父类递归收集 weight/bias
        return {} if result is None else result  # 已绑定：权重由输入 embedding 负责，本层不重复导出（返回容器原样）

    @nvtx_annotate("LMHead")  # NVTX profiling 标记
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：hidden [tokens, dim] -> logits [bs, vocab]；每个 batch 取末 token 的 hidden 计算下一个词分布。"""
        ctx = get_global_ctx()  # 取全局上下文（当前 batch 与 attention 元数据）
        batch = ctx.batch  # 当前调度的批次
        bs = batch.size  # 批内请求数
        if batch.is_prefill:  # prefill 阶段：x 含全部 prompt token 的 hidden，只需每个请求的最后一个
            indices = batch.attn_metadata.get_last_indices(bs)  # 取每个序列最后一个 token 在扁平 token 维上的索引
            x = x[indices].contiguous()  # 按索引抽取 [bs, dim] 并保证内存连续（后续 all_gather 要求）
            del indices  # 及时释放索引张量的引用，降低显存峰值

        module = self.tied_embedding or self  # tie 时借用输入 embedding 的权重；否则用自身
        logits = F.linear(x, module.weight, self.bias)  # 矩阵乘：[bs, dim] @ [dim, vocab_tp]^T -> 每 rank 输出 [bs, vocab_tp] 分片
        if self.tp_size == 1:  # 单卡：分片即完整 logits
            return logits  # 直接返回
        input_shape = logits.shape  # 记录分片形状 (bs, vocab_tp)，供后面还原与拼接
        output_tensor = self._comm.all_gather(logits)  # all_gather：把各 rank 的 [bs, vocab_tp] 拼成 [(bs*vocab_tp)*tp] 一维张量

        if bs == 1:  # 单请求快速路径：无需处理 batch 维拼接
            return output_tensor.view(1, -1)[:, : self.num_embeddings]  # 一维拼直接 reshape 成 [1, vocab*tp]，裁掉词表 padding 部分返回

        output_tensor = output_tensor.view((self.tp_size,) + input_shape)  # 把一维拼接还原为 [tp, bs, vocab_tp]：gather 后按 rank 排列
        output_tensor = output_tensor.permute(1, 0, 2).contiguous()  # 交换为 [bs, tp, vocab_tp]：让同一请求的各 rank 分片相邻（需物理拷贝保证后续 reshape 正确）
        output_tensor = output_tensor.reshape(input_shape[:1] + (self.tp_size * input_shape[1],))  # 合并后两维 -> [bs, vocab*tp]（含 padding）
        return output_tensor[:, : self.num_embeddings]  # 裁掉每行末尾的词表 padding，返回完整 logits [bs, 真实词表大小]
