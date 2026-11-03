from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解使用前向引用（如 torch.Tensor 的延迟导入）

from abc import ABC, abstractmethod  # 导入抽象基类支持：ABC 为基类，abstractmethod 用于声明必须由子类实现的方法
from typing import TYPE_CHECKING  # 类型检查开关：仅静态分析时为 True，运行时为 False，用于避免运行时的循环导入/重量级导入

from minisgl.layers import BaseOP  # 导入 BaseOP：本引擎所有"算子/层"的基类（提供统一参数管理与 forward 约定）

if TYPE_CHECKING:  # 仅在类型检查（如 IDE、mypy）时导入 torch，运行时不加载，减少启动开销
    import torch


class BaseLLMModel(ABC, BaseOP):
    """LLM 抽象基类：所有具体模型（Llama/Qwen/Mistral 等）的统一接口。

    继承 ABC（强制子类实现抽象方法）与 BaseOP（获得层/算子的公共基础设施）。
    """

    @abstractmethod  # 声明抽象方法：子类必须实现，否则实例化时抛 TypeError
    def forward(self) -> torch.Tensor: ...  # 前向计算入口：无显式参数，输入从全局上下文（get_global_ctx）取 batch；返回 logits 张量
