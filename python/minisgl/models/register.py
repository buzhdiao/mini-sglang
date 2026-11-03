import importlib  # 标准库动态导入模块：实现模型类的"懒加载"（用到哪个模型才 import 哪个模块）

from .config import ModelConfig  # 模型配置类：get_model_class 的参数类型

# 模型注册表：HF config.json 的 architectures 类名 → (模块相对路径, 类名) 映射。
# 惰性设计：注册表只存"名字对"，不真正 import，避免加载无关模型拖慢启动。
_MODEL_REGISTRY = {
    "LlamaForCausalLM": (".llama", "LlamaForCausalLM"),  # Llama 系列（含多数微调变体）→ llama.py
    "Qwen2ForCausalLM": (".qwen2", "Qwen2ForCausalLM"),  # Qwen2 系列 → qwen2.py
    "Qwen3ForCausalLM": (".qwen3", "Qwen3ForCausalLM"),  # Qwen3 稠密系列 → qwen3.py
    "Qwen3MoeForCausalLM": (".qwen3_moe", "Qwen3MoeForCausalLM"),  # Qwen3-MoE 系列 → qwen3_moe.py
    "MistralForCausalLM": (".mistral", "MistralForCausalLM"),  # Mistral 系列 → mistral.py
    "Mistral3ForConditionalGeneration": (".mistral", "MistralForCausalLM"),  # Mistral3（多模态入口类）复用 mistral.py 的文本模型
}


def get_model_class(model_architecture: str, model_config: ModelConfig):
    """按架构名查注册表，动态导入并实例化模型类（懒加载工厂函数）。

    参数 model_architecture 是 HF config["architectures"][0]，
    返回的是已实例化的模型对象（而非类本身，函数名略有误导性）。
    """
    if model_architecture not in _MODEL_REGISTRY:  # 架构名未注册：不支持该模型
        raise ValueError(f"Model architecture {model_architecture} not supported")  # 报错并给出架构名，提示用户该模型不在支持列表
    module_path, class_name = _MODEL_REGISTRY[model_architecture]  # 解包：相对模块路径（如 ".llama"）与目标类名
    module = importlib.import_module(module_path, package=__package__)  # 动态导入模块：以当前包为基准解析相对路径，仅此时才真正加载模型代码
    model_cls = getattr(module, class_name)  # 从模块中按名字取出模型类
    return model_cls(model_config)  # 用模型配置实例化并返回（如 LlamaForCausalLM(config)）


__all__ = ["get_model_class"]  # 模块对外只暴露工厂函数，注册表为内部实现细节
