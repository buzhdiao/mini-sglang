# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# ABC/abstractmethod 定义注意力后端的抽象接口
from abc import ABC, abstractmethod
# dataclass 简化纯数据元数据类的声明
from dataclasses import dataclass
# TYPE_CHECKING 仅类型检查时导入；List 用于注解
from typing import TYPE_CHECKING, List

# 仅类型检查时导入 torch 与 Batch，避免运行时循环依赖
if TYPE_CHECKING:
    import torch
    from minisgl.core import Batch


# 注意力元数据抽象基类：每个后端自行定义自己的 metadata 结构
@dataclass
class BaseAttnMetadata(ABC):
    # 抽象方法：返回每个请求本次 forward 中最后一个 token 在 q 序列里的下标，
    # 用于采样阶段只取"最新 token"的 hidden state
    @abstractmethod
    def get_last_indices(self, bs: int) -> torch.Tensor: ...


# 注意力后端抽象基类：统一 prefill/decode/CUDA Graph 相关接口
class BaseAttnBackend(ABC):
    # 核心前向：给定 q/k/v、层号与批次信息，返回注意力输出；
    # 内部负责把 k/v 写入 KV cache 并用分页缓存做 attention
    @abstractmethod
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor: ...

    # 每次调度后根据批次内容构造/刷新本后端需要的元数据（序列长度、页表等）
    @abstractmethod
    def prepare_metadata(self, batch: Batch) -> None: ...

    # CUDA Graph 捕获前的初始化：记录最大序列长度与要捕获的批量列表
    @abstractmethod
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None: ...

    # 进入某个批量 bs 的捕获流程：为该 bs 准备静态（占位）元数据
    @abstractmethod
    def prepare_for_capture(self, batch: Batch) -> None: ...

    # 回放 CUDA Graph 前：把真实批量的动态数据拷进捕获时的静态缓冲区
    @abstractmethod
    def prepare_for_replay(self, batch: Batch) -> None: ...


# 混合后端：prefill 与 decode 可分别使用不同注意力内核实现
class HybridBackend(BaseAttnBackend):
    # 保存两个子后端
    def __init__(
        self,
        prefill_backend: BaseAttnBackend,
        decode_backend: BaseAttnBackend,
    ) -> None:
        # prefill 阶段（长 q）使用的后端
        self.prefill_backend = prefill_backend
        # decode 阶段（q 长度为 1）使用的后端
        self.decode_backend = decode_backend

    # 按批次类型把 forward 委派给对应子后端
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
    ) -> torch.Tensor:
        # prefill 批走 prefill 后端，否则走 decode 后端
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        # 调用子后端的注意力计算
        return backend.forward(q, k, v, layer_id, batch)

    # 同样按批次类型委派元数据准备
    def prepare_metadata(self, batch: Batch) -> None:
        # 选择与批次类型匹配的子后端
        backend = self.prefill_backend if batch.is_prefill else self.decode_backend
        # 由子后端生成自己的 metadata
        return backend.prepare_metadata(batch)

    # CUDA Graph 只用于 decode，因此捕获初始化只交给 decode 后端
    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        # 转发给 decode 后端
        self.decode_backend.init_capture_graph(max_seq_len, bs_list)

    # 捕获阶段的静态元数据也只与 decode 有关
    def prepare_for_capture(self, batch: Batch) -> None:
        # 转发给 decode 后端
        self.decode_backend.prepare_for_capture(batch)

    # 回放前的数据拷贝同样由 decode 后端处理
    def prepare_for_replay(self, batch: Batch) -> None:
        # 转发给 decode 后端
        self.decode_backend.prepare_for_replay(batch)
