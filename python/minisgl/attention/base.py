from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如 torch）

from abc import ABC, abstractmethod  # ABC：抽象基类支持；abstractmethod：声明必须由子类实现的抽象方法
from dataclasses import dataclass  # dataclass 装饰器，自动生成 __init__/__repr__ 等
from typing import TYPE_CHECKING, List  # TYPE_CHECKING：仅类型检查时导入；List：泛型列表注解

if TYPE_CHECKING:  # 仅静态类型检查时导入，运行时不执行（避免循环依赖与启动开销）
    import torch  # torch 仅用于类型注解（forward 的参数/返回值类型）
    from minisgl.core import Batch  # 核心批次数据结构，仅用于类型注解


@dataclass  # 数据类装饰器，自动生成 __init__/__repr__ 等方法
class BaseAttnMetadata(ABC):
    """注意力元数据抽象基类：封装一次前向所需的索引/长度信息（由 prepare_metadata 构建）。

    各后端（FA/FlashInfer/TRT-LLM）各自派生出具体元数据类，字段与底层 kernel 的入参对应。
    """

    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...
    # 抽象方法：返回批内每个请求"最后一个 token"在扁平 q 序列中的位置索引（形状 [bs]）。
    # 用于采样阶段从变长 prefill 输出中取每个请求最后一个位置的 hidden state；bs 为真实请求数。


class BaseAttnBackend(ABC):
    """注意力后端抽象基类：统一封装不同注意力实现（FlashAttention/FlashInfer/TRT-LLM）的调用接口。

    生命周期：prepare_metadata（每次前向前构建元数据）→ forward（执行注意力计算）；
    CUDA Graph 相关：init_capture_graph → prepare_for_capture（录制）→ prepare_for_replay（重放）。
    """

    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...
    # 抽象方法：执行一层注意力的完整计算。
    # q/k/v 为当前层的 QKV 投影输出；layer_id 用于定位该层的 KV Cache 切片；
    # 先把 k/v 写入 paged KV Cache，再用后端专属 kernel 做 attention，返回注意力输出。

    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...
    # 抽象方法：根据 batch 中各请求的长度信息，预先构建本次前向所需的注意力元数据
    # （累积长度 cu_seqlens、页表 page_table 等），结果挂到 batch.attn_metadata 上。

    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...
    # 抽象方法：初始化 CUDA Graph 捕获：为最大 batch size 分配捕获期元数据缓冲区，
    # bs_list 是计划捕获的所有 batch size（decode 阶段每个 bs 录一张图）。

    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...
    # 抽象方法：进入 CUDA Graph 录制前调用：基于捕获期静态缓冲区构造元数据并挂到 batch 上。

    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...
    # 抽象方法：CUDA Graph 重放前调用：把真实 batch 的元数据拷贝进捕获期静态缓冲区（原地 copy_，地址不变），保证重放读到新数据。


class HybridBackend(BaseAttnBackend):
    """混合注意力后端：prefill 与 decode 阶段可分别使用不同的注意力实现。

    例如 "fa,trtllm"：prefill 用 FlashAttention，decode 用 TRT-LLM kernel，
    各阶段特点不同（prefill 是变长扩展计算、decode 是单 token 生成），分开选型可各取所长。
    """

    def __init__(
        self,
        prefill_backend: BaseAttnBackend,  # prefill（预填充/扩展）阶段使用的后端
        decode_backend: BaseAttnBackend,  # decode（逐 token 解码）阶段使用的后端
    ) -> None:
        self.prefill_backend = prefill_backend  # 保存 prefill 后端实例
        self.decode_backend = decode_backend  # 保存 decode 后端实例

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # 按 batch 当前所处阶段（prefill/decode）选择对应的后端
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.forward(q, k, v, layer_id, batch)  # 委托给选中的后端执行注意力计算

    def prepare_metadata(self, batch: Batch) -> None:
        # 元数据构建同样按阶段分发给对应后端（两者元数据格式不同）
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        return backend.prepare_metadata(batch)  # 委托构建元数据（结果挂在 batch.attn_metadata 上）

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # CUDA Graph 只用于 decode 阶段（prefill 变长、批量小，录图收益低），故只初始化 decode 后端
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    def prepare_for_capture(self, batch: Batch) -> None:
        # 录制仅发生在 decode 阶段，直接委托给 decode 后端
        self.decode_backend.prepare_for_capture(batch)

    def prepare_for_replay(self, batch: Batch) -> None:
        # 重放仅发生在 decode 阶段，直接委托给 decode 后端
        self.decode_backend.prepare_for_replay(batch)
