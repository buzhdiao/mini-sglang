# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# 抽象方法支持
from abc import abstractmethod
# 泛型与类型工具
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar

import torch

# state_dict 类型别名：参数名 -> 张量
_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    # 拼接层级参数名：有前缀时用点连接（如 layers.0.q_proj.weight），否则用原名
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    # 算子基类：不用 torch.nn.Module，改用轻量自定义的 state_dict 机制，
    # 避免注册钩子等开销，同时支持张量与子算子的递归收集
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    # 前向计算接口，由子类实现

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        # 收集本算子的全部参数：递归遍历实例属性
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            # 下划线开头的属性视为私有缓存，不导出
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                # 张量直接登记：参数名 = 前缀.属性名
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                # 子算子递归收集，前缀叠加子算子名
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        # 从 state_dict 加载参数：按属性名逐个取出并做形状/类型校验
        for name, param in self.__dict__.items():
            # 私有属性跳过
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                # pop 掉已消费的键，便于最后检测多余键
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                # 形状与 dtype 必须完全一致才允许赋值
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                # 子算子递归加载；_internal=True 表示仍在递归中
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            # 递归结束后仍有剩余键：权重文件与模型结构不匹配
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")


class StateLessOP(BaseOP):
    # 无参数算子基类：state_dict 恒为空（如注意力调度、RoPE 等纯计算算子）
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        # 无参数算子不应接收任何键；顶层调用时若 state_dict 非空则报错
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        # 无参数可导出，直接透传/新建空结果
        return result if result is not None else {}


# 泛型参数：限定 OPList 内元素必须是 BaseOP
T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    # 算子列表：把一组同型算子（如 Transformer 的各层）聚合成一个复合算子，
    # 参数名以数字下标区分（如 layers.0.xxx / layers.1.xxx）
    def __init__(self, ops: List[T]):
        super().__init__()
        self.op_list = ops

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        # 逐个子算子收集，前缀加上其下标
        result = result if result is not None else {}
        for i, op in enumerate(self.op_list):
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        # 逐个子算子加载（内部递归不检查剩余键）
        for i, op in enumerate(self.op_list):
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            # 顶层调用结束后仍有多余键则报错
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
