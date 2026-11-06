# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

from typing import Any, Dict, Type

import numpy as np
import torch


def _serialize_any(value: Any) -> Any:
    # 递归序列化任意值：容器逐元素处理，标量直接透传，其余按 dataclass 序列化
    if isinstance(value, dict):
        # 字典：键保持原样，值递归序列化
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        # 列表/元组：保留原容器类型，递归处理元素
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):
        # JSON 原生可表示的标量直接透传
        return value
    else:
        # 其他对象（dataclass、张量等）走类型化序列化
        return serialize_type(value)


def serialize_type(self) -> Dict:
    # find all member variables
    # 把对象序列化成带类型标签的字典：__type__ 记录类型名，反序列化时据此重建
    serialized = {}

    if isinstance(self, torch.Tensor):
        # 张量走二进制路径：保留零拷贝转 bytes，避免逐元素 JSON 化的巨大开销
        assert self.dim() == 1, "we can only serialize 1D tensor for now"
        serialized["__type__"] = "Tensor"
        # 张量数据转成原始字节串（如 int32 token id 序列）
        serialized["buffer"] = self.numpy().tobytes()
        # 记录 dtype 字符串，反序列化时还原
        serialized["dtype"] = str(self.dtype)
        return serialized

    # normal type
    # 普通类型：登记类名，再逐个成员变量序列化
    serialized["__type__"] = self.__class__.__name__
    for k, v in self.__dict__.items():
        serialized[k] = _serialize_any(v)
    return serialized


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    # 递归反序列化：与 _serialize_any 对称
    if isinstance(data, dict):
        if "__type__" in data:
            # 带类型标签的字典：走类型化重建
            return deserialize_type(cls_map, data)
        else:
            # 普通字典：逐值递归
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        # 列表/元组：保留原容器类型递归还原
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):
        # 标量直接透传
        return data
    else:
        # 其余类型不应出现在序列化结果里
        raise ValueError(f"Cannot deserialize type {type(data)}")


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    # 按类型标签重建对象：cls_map 通常传入消息模块的 globals() 以查找消息类
    type_name = data["__type__"]
    # we can only serialize 1D tensor for now
    if type_name == "Tensor":
        # 张量重建路径：bytes -> numpy -> torch（copy 保证内存独占）
        buffer = data["buffer"]
        # 去掉 "torch." 前缀映射成 numpy dtype 名（如 int32）
        dtype_str = data["dtype"].replace("torch.", "")
        np_dtype = getattr(np, dtype_str)
        assert isinstance(buffer, bytes)
        # 从原始字节按 dtype 解释出一维数组
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)
        # 转 torch 张量；copy 使其拥有独立内存（frombuffer 是只读视图）
        return torch.from_numpy(np_tensor.copy())

    # 按 cls_map 查找类型并构造实例
    cls = cls_map[type_name]
    kwargs = {}
    for k, v in data.items():
        # 跳过类型标签本身
        if k == "__type__":
            continue
        # 其余字段递归反序列化成构造参数
        kwargs[k] = _deserialize_any(cls_map, v)
    return cls(**kwargs)
