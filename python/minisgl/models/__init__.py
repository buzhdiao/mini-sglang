# 模型抽象基类
from .base import BaseLLMModel
# 模型配置与 RoPE 配置
from .config import ModelConfig, RotaryConfig
# 按架构名查找模型类的注册器
from .register import get_model_class
# HF 权重加载
from .weight import load_weight


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    # 按配置里的架构名（如 LlamaForCausalLM）实例化对应模型
    return get_model_class(model_config.architectures[0], model_config)


__all__ = ["create_model", "load_weight", "RotaryConfig"]
