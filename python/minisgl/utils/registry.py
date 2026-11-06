# 泛型工具：注册表可标注条目类型
from typing import Callable, Generic, Iterable, List, TypeVar

# 条目类型参数
T = TypeVar("T")


# 通用注册表：按名字注册/查找实现类（注意力后端、MoE 后端、cache 管理器等）
class Registry(Generic[T]):
    def __init__(self, type: str):
        # 名字 -> 条目的映射表
        self._registry = {}
        # 注册表类别名（仅用于报错信息，如 "attention backend"）
        self._type = type

    def register(self, name: str) -> Callable[[T], None]:
        if name in self._registry:
            # 同名重复注册会掩盖实现，直接拒绝
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item: T) -> None:
            # 装饰器本体：把被装饰的类/函数写入注册表
            self._registry[name] = item

        # 返回装饰器（用法：@REGISTRY.register("name")）
        return decorator

    def __getitem__(self, name: str) -> T:
        if name not in self._registry:
            # 未注册的名字给出清晰错误
            raise KeyError(f"Unsupported {self._type}: {name}")
        # 按名字取出实现
        return self._registry[name]

    def supported_names(self) -> List[str]:
        # 列出全部已注册名字（常用于 argparse choices）
        return list(self._registry.keys())

    def assert_supported(self, names: str | Iterable[str]) -> None:
        if isinstance(names, str):
            # 单个名字统一成列表处理
            names = [names]
        for name in names:
            if name not in self._registry:
                from argparse import ArgumentTypeError

                # 抛参数错误：argparse 能把它转成友好的命令行报错
                raise ArgumentTypeError(
                    f"Unsupported {self._type}: {name}. "
                    f"Supported items: {self.supported_names()}"
                )
