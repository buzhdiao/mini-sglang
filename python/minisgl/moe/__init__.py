from __future__ import annotations  # 启用延迟注解求值，允许类型注解使用前向引用

from typing import Protocol  # Protocol：结构化子类型（静态鸭子类型），只要求"有该方法"不要求继承

from minisgl.utils import Registry, init_logger  # Registry：通用名字→对象注册表；init_logger：按模块名创建标准 logger

from .base import BaseMoeBackend  # MoE 后端抽象基类：定义 forward 统一签名

logger = init_logger(__name__)  # 创建本模块 logger（名字如 minisgl.moe），便于日志追踪


class MoeBackendCreator(Protocol):
    """MoE 后端工厂的协议约束：任何"无参可调用、返回 BaseMoeBackend"的对象都符合此协议。

    用途：注册表中存放的创建函数都满足该签名，取出后直接调用即可获得后端实例。
    """

    def __call__(self) -> BaseMoeBackend: ...  # 协议方法声明：调用后返回一个 MoE 后端实例（无具体实现）


SUPPORTED_MOE_BACKENDS = Registry[MoeBackendCreator]("MoE Backend")  # 已支持后端的注册表：名字 → 工厂函数（如 "fused" → create_fused_moe_backend）


@SUPPORTED_MOE_BACKENDS.register("fused")  # 将下方函数以 "fused" 为名注册进注册表（装饰器返回原函数，不改变行为）
def create_fused_moe_backend():
    """创建 fused MoE 后端的工厂函数。

    import 放在函数体内（延迟导入）：只有真正选用 fused 后端时才加载其依赖，
    避免未用 MoE 的场景付出 import 开销，也规避潜在循环导入。
    """
    from .fused import FusedMoe  # 延迟导入 fused 实现（top-k 路由 + Triton fused kernel）

    return FusedMoe()  # 实例化并返回后端对象


def create_moe_backend(backend: str) -> BaseMoeBackend:
    """按名字创建 MoE 后端：从注册表取出工厂函数并调用（未知名字由 Registry 报错）。"""
    return SUPPORTED_MOE_BACKENDS[backend]()  # 字典式取用注册表项得到工厂函数，紧跟 () 立即调用生成后端


__all__ = [  # 模块对外暴露的公共接口
    "BaseMoeBackend",  # 后端基类（供自定义后端继承）
    "create_moe_backend",  # 后端工厂入口
    "SUPPORTED_MOE_BACKENDS",  # 注册表（供其他模块 @register 新后端）
]
