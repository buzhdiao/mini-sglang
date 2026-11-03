from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如未导入的 ModelConfig）

from typing import TYPE_CHECKING, Protocol  # TYPE_CHECKING：仅类型检查时导入；Protocol：结构化子类型（鸭子类型的静态表达）

from minisgl.utils import Registry, init_logger  # Registry：按名字注册/查找的通用注册表；init_logger：初始化日志器

from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend  # 注意力后端的抽象基类与混合后端实现

if TYPE_CHECKING:  # 仅在静态类型检查（如 mypy/IDE）时导入，运行时不执行，避免循环依赖/启动开销
    from minisgl.models import ModelConfig  # 模型配置，仅用于类型注解

logger = init_logger(__name__)  # 创建本模块的 logger，日志会带上模块路径前缀


class BackendCreator(Protocol):  # 后端工厂函数的"协议"（结构化接口）：任何可调用对象只要签名匹配即视为该类型
    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...  # 工厂签名：接收模型配置，返回一个注意力后端实例


# 注意力后端注册表：全局单例，各后端模块通过装饰器把工厂函数注册进来，键为后端名（如 "fa"/"fi"/"trtllm"）
SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


@SUPPORTED_ATTENTION_BACKENDS.register("trtllm")  # 把下面的工厂函数注册为 "trtllm" 后端
def create_trtllm_backend(config: ModelConfig):
    # 延迟导入 TensorRTLLMBackend：只有真正选用该后端时才加载其依赖（flashinfer），避免启动时不必要的导入
    from .trtllm import TensorRTLLMBackend

    return TensorRTLLMBackend(config)  # 用模型配置实例化 TRT-LLM 注意力后端


@SUPPORTED_ATTENTION_BACKENDS.register("fi")  # 把下面的工厂函数注册为 "fi"（FlashInfer）后端
def create_fi_backend(config: ModelConfig):
    from .fi import FlashInferBackend  # 延迟导入 FlashInfer 后端实现

    return FlashInferBackend(config)  # 实例化 FlashInfer 后端


@SUPPORTED_ATTENTION_BACKENDS.register("fa")  # 把下面的工厂函数注册为 "fa"（FlashAttention）后端
def create_fa_backend(config: ModelConfig):
    from .fa import FlashAttentionBackend  # 延迟导入 FlashAttention 后端实现

    return FlashAttentionBackend(config)  # 实例化 FlashAttention 后端


def validate_attn_backend(backend: str, allow_auto: bool = True):
    """校验注意力后端名字字符串是否合法（是否都已在注册表中注册）。

    backend 形如 "fa"、"fi" 或混合形式 "fa,trtllm"（逗号分隔的 P/D 两阶段后端）；
    "auto" 是特殊值，表示自动选择，仅当 allow_auto=True 时才被接受。
    """
    if backend != "auto":  # 非 "auto" 时需要逐一校验每个后端名
        # 混合后端形如 "fa,trtllm"，按逗号拆成列表；单后端则包装成单元素列表
        required_backends = backend.split(",") if "," in backend else [backend]
        # 断言所有后端名都已注册，否则抛错提示用户选择了不支持的后端
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(required_backends)
    else:
        # "auto" 只在允许自动选择时合法（如启动参数可用 auto，但内部创建后端时必须显式指定）
        assert allow_auto, "auto is not allowed here"
    return backend  # 原样返回校验通过的 backend 字符串


def create_attention_backend(
    backend: str,
    config: ModelConfig,
) -> BaseAttnBackend:
    """根据后端名字符串创建对应的注意力后端实例。

    支持三种形式：
    1. 单后端："fa"/"fi"/"trtllm"
    2. 混合后端："prefill后端,decode后端"（如 "fa,trtllm"），P/D 两阶段可用不同实现
    3. 相同的 P/D 后端："fa,fa" 会退化为单后端
    """
    validate_attn_backend(backend, allow_auto=False)  # 先校验合法性；此处不允许 "auto"，必须显式指定
    if "," in backend:  # 包含逗号说明是混合后端（prefill,decode 两段式）
        # 混合形式只允许一个逗号（即恰好两段），更多逗号视为配置错误
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        # 按第一个逗号拆分为 prefill 后端名与 decode 后端名
        p_backend, d_backend = backend.split(",", 1)
        if p_backend != d_backend:  # 两段不同：真正需要混合后端
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")  # 记录混合后端选择
            p_backend = create_attention_backend(p_backend, config)  # 递归创建 prefill 阶段的后端实例
            d_backend = create_attention_backend(d_backend, config)  # 递归创建 decode 阶段的后端实例
            return HybridBackend(p_backend, d_backend)  # 用 HybridBackend 包装：按 batch 阶段动态分发
        backend = p_backend  # both are the same, fall through to single backend  # 两段相同：退化为单后端，继续走统一创建路径
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")  # 提示用户无需混合

    # 从注册表取出工厂函数并调用，返回单后端实例（注意：此时 backend 可能已被上面的退化逻辑改写）
    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


__all__ = [  # 模块公开接口：`from minisgl.attention import *` 时只导出这些名字
    "BaseAttnMetadata",  # 注意力元数据抽象基类
    "BaseAttnBackend",  # 注意力后端抽象基类
    "create_attention_backend",  # 后端工厂入口
    "SUPPORTED_ATTENTION_BACKENDS",  # 后端注册表
    "validate_attn_backend",  # 后端名校验函数
]
