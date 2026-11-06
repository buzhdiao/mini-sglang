# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# Protocol 定义 MoE 后端工厂的结构化类型
from typing import Protocol

# 注册表与日志工具
from minisgl.utils import Registry, init_logger

from .base import BaseMoeBackend

logger = init_logger(__name__)


class MoeBackendCreator(Protocol):
    # 后端工厂协议：无参调用，返回一个 MoE 后端实例
    def __call__(self) -> BaseMoeBackend: ...


# 已支持的 MoE 后端注册表：键为后端名（如 "fused"）
SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")


# 注册 "fused" 后端：路由 + 分组 GEMM 全部在融合核里完成
@SUPPORTED_MOE_BACKENDS.register("fused")
def create_fused_moe_backend():
    # 延迟导入，避免包加载时引入 Triton 编译依赖
    from .fused import FusedMoe

    return FusedMoe()


def create_moe_backend(backend: str) -> BaseMoeBackend:
    # 按名字从注册表实例化 MoE 后端
    return SUPPORTED_MOE_BACKENDS[backend]()


__all__ = [
    "BaseMoeBackend",
    "create_moe_backend",
    "SUPPORTED_MOE_BACKENDS",
]
