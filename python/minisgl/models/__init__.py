from .base import BaseLLMModel  # 导入所有因果语言模型（LLM）的抽象基类
from .config import ModelConfig, RotaryConfig  # 导入模型配置：ModelConfig（模型结构超参集合）与 RotaryConfig（RoPE 旋转位置编码配置）
from .register import get_model_class  # 导入模型类获取函数：根据 HF config 的 architectures 字段按需加载对应模型类
from .weight import load_weight  # 导入权重加载器：流式读取 safetensors 并做 TP 切分与投影合并


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    """模型工厂函数：根据配置中的 architectures 字段（HF config.json 里的模型类名）
    查注册表并实例化对应模型对象（内部通过懒加载 import 避免一次性导入全部模型）。"""
    return get_model_class(model_config.architectures[0], model_config)  # 取第一个架构名（通常只填一个），传入配置实例化模型


__all__ = ["create_model", "load_weight", "RotaryConfig"]  # 模块对外暴露的公共接口
