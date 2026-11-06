# importlib 用于按需延迟导入模型模块
import importlib

from .config import ModelConfig

# HF 架构类名 -> (模块路径, 类名) 的注册表；
# 延迟导入避免包加载时把所有模型代码都拉起来
_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),
    # Mistral3 多模态模型复用文本侧的 Mistral 实现
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    # 按架构名实例化模型：查表 -> 导入模块 -> 取类 -> 用配置构造实例
    if model_architecture not in _MODEL_REGISTRY:
        raise ValueError(f"Model architecture {model_architecture} not supported")
    module_path, class_name = _MODEL_REGISTRY[model_architecture]
    # 相对导入当前包下的模型模块
    module = importlib.import_module(module_path, package=__package__)
    model_cls = getattr(module, class_name)
    return model_cls(model_config)


__all__ = ["get_model_class"]