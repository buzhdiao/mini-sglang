from typing import Callable, Generic, Iterable, List, TypeVar  # 类型工具：把注册器泛型化，便于类型检查

T = TypeVar("T")  # 被注册对象（类/函数）的类型占位符


class Registry(Generic[T]):
    """通用名字注册器：模块导入时用装饰器注册实现，运行时按名字查找。
    典型用途：注册注意力后端 / 模型架构 / 分词器等可插拔组件。
    """

    def __init__(self, type: str):  # type：注册对象的类别展示名（如 "attention"），仅用于拼接错误信息
        self._registry = {}  # 名字 -> 对象 的存储字典
        self._type = type  # 保存类别名（注意：参数名遮蔽了内建 type，仅局部作用域）

    def register(self, name: str) -> Callable[[T], None]:
        """注册装饰器工厂：先查重，再返回把对象写入注册表的装饰器"""
        if name in self._registry:  # 同名重复注册说明代码冲突，直接报错尽早暴露
            raise KeyError(f"{self._type} '{name}' is already registered.")

        def decorator(item: T) -> None:
            self._registry[name] = item  # 实际注册动作：记录 name -> item 映射

        return decorator  # 返回装饰器，供 @registry.register("xxx") 语法使用

    def __getitem__(self, name: str) -> T:
        """支持 registry[name] 语法按名字取对象；未注册时抛带类别信息的 KeyError"""
        if name not in self._registry:  # 名字不存在时给出友好的类别上下文
            raise KeyError(f"Unsupported {self._type}: {name}")
        return self._registry[name]  # 返回注册的对象

    def supported_names(self) -> List[str]:
        """列出所有已注册的名字（用于报错提示 / CLI 帮助信息）"""
        return list(self._registry.keys())  # 拷贝一份键列表，避免外部直接改动内部字典

    def assert_supported(self, names: str | Iterable[str]) -> None:
        """校验名字是否已注册；未注册抛 ArgumentTypeError，便于 argparse 直接给出标准错误提示"""
        if isinstance(names, str):  # 兼容单个字符串入参：包装成列表统一处理
            names = [names]
        for name in names:  # 逐个校验
            if name not in self._registry:  # 发现未注册项即失败
                from argparse import ArgumentTypeError  # 延迟导入：仅错误路径才需要

                raise ArgumentTypeError(  # 用 argparse 专用异常类型，CLI 解析阶段能正确捕获并提示
                    f"Unsupported {self._type}: {name}. "
                    f"Supported items: {self.supported_names()}"  # 报错时附上全部可选名字
                )
