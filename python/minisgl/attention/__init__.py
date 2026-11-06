# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# TYPE_CHECKING 仅类型检查时导入；Protocol 定义"工厂函数"的结构类型
from typing import TYPE_CHECKING, Protocol

# Registry 提供按名字注册/查找的注册表；init_logger 创建模块 logger
from minisgl.utils import Registry, init_logger

# 注意力后端的抽象基类、元数据基类与混合后端
from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend

# 仅在静态类型检查时导入模型配置，运行时不产生循环依赖
if TYPE_CHECKING:
    from minisgl.models import ModelConfig

# 创建模块级 logger
logger = init_logger(__name__)


# 工厂协议：任何"接收 ModelConfig 并返回注意力后端"的可调用对象都符合该结构
class BackendCreator(Protocol):
    # 协议只声明签名，不做实现
    def __call__(self, config: ModelConfig) -> BaseAttnBackend: ...


# 注意力后端注册表：字符串名 -> 工厂函数，供 --attn-backend 配置按名创建
SUPPORTED_ATTENTION_BACKENDS = Registry[BackendCreator]("Attention Backend")


# 把 "trtllm" 注册为 TensorRT-LLM 内核后端（flashinfer 的 trtllm decode/prefill 内核）
@SUPPORTED_ATTENTION_BACKENDS.register("trtllm")
def create_trtllm_backend(config: ModelConfig):
    # 延迟导入：只有真正选用该后端时才加载其依赖
    from .trtllm import TensorRTLLMBackend

    # 用模型配置实例化后端
    return TensorRTLLMBackend(config)


# 把 "fi" 注册为 FlashInfer 后端（ragged/分页 KV 的专用 wrapper）
@SUPPORTED_ATTENTION_BACKENDS.register("fi")
def create_fi_backend(config: ModelConfig):
    # 延迟导入，避免未选用的后端拖慢启动
    from .fi import FlashInferBackend

    # 用模型配置实例化后端
    return FlashInferBackend(config)


# 把 "fa" 注册为 FlashAttention 后端（sgl-kernel 的 flash_attn_with_kvcache）
@SUPPORTED_ATTENTION_BACKENDS.register("fa")
def create_fa_backend(config: ModelConfig):
    # 延迟导入，避免未选用的后端拖慢启动
    from .fa import FlashAttentionBackend

    # 用模型配置实例化后端
    return FlashAttentionBackend(config)


# 校验后端名字字符串是否都受支持；allow_auto 决定是否放行 "auto"（由调用方再决定）
def validate_attn_backend(backend: str, allow_auto: bool = True):
    # 非 auto 时按逗号拆出多个后端名逐一校验
    if backend != "auto":
        # 混合后端形如 "fa,fi"（prefill,decode），单一后端则只有一个元素
        required_backends = backend.split(",") if "," in backend else [backend]
        # 断言每个名字都已在注册表中
        SUPPORTED_ATTENTION_BACKENDS.assert_supported(required_backends)
    else:
        # auto 只在允许的场景（如启动参数默认值）可用，此处不应出现
        assert allow_auto, "auto is not allowed here"
    # 原样返回，便于链式校验
    return backend


# 根据后端名字符串创建注意力后端实例（支持 "prefill,decode" 混合组合）
def create_attention_backend(
    backend: str,
    config: ModelConfig,
) -> BaseAttnBackend:
    # 先做合法性校验（此处不允许 auto）
    validate_attn_backend(backend, allow_auto=False)
    # 含逗号表示要组合 prefill 与 decode 两个后端
    if "," in backend:
        # 只允许一个逗号，即最多两段
        assert backend.count(",") == 1, "Only one comma is allowed in hybrid backend"
        # 拆出 prefill 段与 decode 段
        p_backend, d_backend = backend.split(",", 1)
        # 两段不同才有必要做混合路由
        if p_backend != d_backend:
            # 记录混合后端组合，便于排查
            logger.info(f"Using hybrid attention backend: prefill={p_backend}, decode={d_backend}")
            # 递归创建 prefill 用的后端
            p_backend = create_attention_backend(p_backend, config)
            # 递归创建 decode 用的后端
            d_backend = create_attention_backend(d_backend, config)
            # 用 HybridBackend 按批次类型（prefill/decode）路由到对应实现
            return HybridBackend(p_backend, d_backend)
        # 两者相同则退化为单一后端
        backend = p_backend  # both are the same, fall through to single backend
        # 提示用户配置冗余
        logger.warning(f"P/D attention backends are the same: {backend}, using single backend.")

    # 从注册表按名取工厂函数并实例化后端
    return SUPPORTED_ATTENTION_BACKENDS[backend](config)


# 声明该包对外暴露的公共接口
__all__ = [
    "BaseAttnMetadata",
    "BaseAttnBackend",
    "create_attention_backend",
    "SUPPORTED_ATTENTION_BACKENDS",
    "validate_attn_backend",
]
