# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# 抽象基类与抽象方法
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

# 继承算子基类：获得轻量 state_dict 机制
from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    # 所有语言模型（Llama/Qwen/Mistral 等）的抽象基类：
    # 复用 BaseOP 的参数收集/加载，子类实现具体的 forward
    @abstractmethod
    def forward(self) -> torch.Tensor: ...
    # 前向接口：由各模型子类实现（输入形状/上下文取自全局 ctx）
