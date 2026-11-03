from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如 _STATE_DICT | None）

from abc import abstractmethod  # 导入抽象方法装饰器，用于声明子类必须实现的接口
from typing import Any, Dict, Generic, List, TypeAlias, TypeVar  # 类型工具：Any/Dict/Generic/List 及别名、泛型变量支持

import torch  # PyTorch 主库：本模块用其 Tensor 类型与设备/形状校验

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]  # 类型别名：权重字典，键为带前缀的参数名，值为张量


def _concat_prefix(prefix: str, name: str) -> str:
    """拼接层级参数名前缀，如 ("layers.0", "attn") -> "layers.0.attn"；前缀为空时直接返回参数名"""
    return f"{prefix}.{name}" if prefix else name  # 用点号连接形成 HF 权重字典中的层级键名


class BaseOP:
    """所有模型层算子的基类：自定义轻量"模块"协议（state_dict 递归收集 + load 校验）。

    与 torch.nn.Module 不同，不注册参数/不追踪 hook，直接遍历 __dict__ 中的
    Tensor 与嵌套 BaseOP，实现极简的权重存取。
    """

    @abstractmethod  # 抽象方法装饰器：强制子类实现 forward，否则实例化报错
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...  # 算子前向计算入口，签名由具体子类定义

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        """递归收集本算子（含子算子）的全部张量，键名带层级前缀。

        prefix: 当前递归的键名前缀；result: 复用的输出字典（跨层收集时传入共享容器）。
        """
        result = result if result is not None else {}  # 未提供结果容器则新建；提供则原地写入以便上层聚合

        for name, param in self.__dict__.items():  # 遍历实例属性（不依赖 nn.Module 的 _parameters 注册机制）
            if name.startswith("_"):  # 以下划线开头的属性视为私有（如 _comm 通信器），跳过
                continue  # 私有属性不属于模型权重
            if isinstance(param, torch.Tensor):  # 张量属性即模型参数
                result[_concat_prefix(prefix, name)] = param  # 以 "前缀.属性名" 为键存入结果字典（存引用而非拷贝）
            elif isinstance(param, BaseOP):  # 嵌套的子算子，递归向下收集
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)  # 子算子键名加上自身属性名前缀

        return result  # 返回收集好的权重字典

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,  # 外部权重字典（如 HF checkpoint），加载成功后会从中 pop 消耗掉键
        *,
        prefix: str = "",  # 当前层级的键名前缀
        _internal: bool = False,  # 内部递归标志：子算子调用时为 True，抑制顶层"多余键"报错
    ) -> None:
        """从权重字典加载参数：pop 对应键、校验形状/dtype 后替换本对象的张量属性。"""
        for name, param in self.__dict__.items():  # 遍历实例属性，寻找需要加载的张量与子算子
            if name.startswith("_"):  # 跳过私有属性
                continue  # 私有属性不参与权重加载
            if isinstance(param, torch.Tensor):  # 目标张量属性：从字典中取出对应权重
                item = state_dict.pop(_concat_prefix(prefix, name))  # pop 取出并从字典删除，剩余键即"多余键"
                assert isinstance(item, torch.Tensor)  # 断言加载项确为张量，防止字典值类型错误
                assert param.shape == item.shape and param.dtype == item.dtype  # 校验形状与 dtype 完全一致，防止静默错载
                setattr(self, name, item)  # 用加载的张量替换原属性（原 empty 占位张量被丢弃）
            elif isinstance(param, BaseOP):  # 嵌套子算子：递归加载
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True  # 传递拼接前缀并标记为内部递归
                )

        if not _internal and state_dict:  # 仅在顶层调用时检查：字典若仍有剩余键说明加载了多余权重
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")  # 报错指出未匹配的键，帮助发现配置不一致


class StateLessOP(BaseOP):
    """无状态算子基类：不含任何可加载权重（如 RoPE、AttentionLayer 封装）。

    state_dict/load_state_dict 被短路为空操作，但仍校验顶层不能有多余键。
    """

    def __init__(self):
        super().__init__()  # 调用父类构造（BaseOP 无自定义构造，仅保持继承链完整）

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,  # 权重字典（无状态算子不消费任何键）
        *,
        prefix: str = "",  # 键名前缀（此处仅用于报错信息）
        _internal: bool = False,  # 内部递归标志
    ) -> None:
        if not _internal and state_dict:  # 顶层调用且字典非空：无状态算子不应接收任何权重
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")  # 多余键报错，帮助发现权重文件与模型结构不匹配

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return result if result is not None else {}  # 无权重可收集：原样返回传入容器或空字典


T = TypeVar("T", bound=BaseOP)  # 泛型变量：约束 OPList 内元素类型必须是 BaseOP 子类


class OPList(BaseOP, Generic[T]):
    """算子列表：按索引为子算子的权重键名编号（如 "layers.0"、"layers.1"）。

    用于组织 transformer 的多层结构，泛型参数 T 保留元素静态类型信息。
    """

    def __init__(self, ops: List[T]):
        super().__init__()  # 调用父类构造
        self.op_list = ops  # 保存子算子列表（注意：张量属性存于各子算子内，本层自身无权重）

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}  # 复用或新建结果容器
        for i, op in enumerate(self.op_list):  # 遍历子算子并带上索引
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)  # 以列表索引作为键名层级（"前缀.0/1/2..."）递归收集
        return result  # 返回聚合结果

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,  # 权重字典
        *,
        prefix: str = "",  # 键名前缀
        _internal: bool = False,  # 内部递归标志
    ) -> None:
        for i, op in enumerate(self.op_list):  # 逐个加载子算子
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)  # 键名规则与 state_dict 对称，标记内部递归

        if not _internal and state_dict:  # 顶层调用时的多余键校验（与 BaseOP.load_state_dict 尾部逻辑一致）
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")  # 报错指出未被任何子算子消费的键
