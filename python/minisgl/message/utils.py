from __future__ import annotations  # 启用延迟注解求值，允许前向引用类型

from typing import Any, Dict, Type  # 类型注解工具：Any/Dict/Type

import numpy as np  # NumPy：用于张量字节级序列化（dtype 转换与 frombuffer）
import torch  # PyTorch：Tensor 类型的识别与重建


def _serialize_any(value: Any) -> Any:
    """递归序列化任意值：容器逐元素展开，基本类型原样保留，其余对象走 serialize_type"""
    if isinstance(value, dict):  # 字典：逐键值递归
        return {k: _serialize_any(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):  # 列表/元组：保持原类型递归展开
        return type(value)(_serialize_any(v) for v in value)
    elif isinstance(value, (int, float, str, type(None), bool, bytes)):  # JSON 可直接表达的基本类型
        return value  # 原样保留，无需处理
    else:
        return serialize_type(value)  # 其他对象（如嵌套消息）递归走对象序列化


def serialize_type(self) -> Dict:
    """把对象序列化为带 __type__ 类型标记的字典；Tensor 走专门的字节级路径"""
    # find all member variables  # 遍历全部成员变量
    serialized = {}  # 输出字典

    if isinstance(self, torch.Tensor):  # 张量特殊处理：转原始字节传输
        assert self.dim() == 1, "we can only serialize 1D tensor for now"  # 目前仅支持一维张量
        serialized["__type__"] = "Tensor"  # 类型标记为 Tensor，反序列化时走还原分支
        serialized["buffer"] = self.numpy().tobytes()  # 转为 NumPy 再取原始字节（可直接塞进 JSON/网络帧）
        serialized["dtype"] = str(self.dtype)  # 记录 dtype 字符串（如 "torch.int32"）供还原
        return serialized  # 张量分支到此结束

    # normal type  # 普通对象：记录类名并逐字段递归序列化
    serialized["__type__"] = self.__class__.__name__  # 类名作为还原依据（decoder 侧按名查类）
    for k, v in self.__dict__.items():  # 遍历实例的 __dict__
        serialized[k] = _serialize_any(v)  # 每个字段递归处理（字段可能还是对象/张量/容器）
    return serialized  # 返回可传输的字典


def _deserialize_any(cls_map: Dict[str, Type], data: Any) -> Any:
    """递归反序列化：容器逐元素还原，带 __type__ 的字典还原为对象，基本类型原样返回"""
    if isinstance(data, dict):  # 字典：可能是对象标记，也可能是普通字典
        if "__type__" in data:  # 含类型标记：是序列化后的对象
            return deserialize_type(cls_map, data)  # 还原为真正的类实例
        else:
            return {k: _deserialize_any(cls_map, v) for k, v in data.items()}  # 普通字典：逐值递归
    elif isinstance(data, (list, tuple)):  # 列表/元组：保持原类型递归还原
        return type(data)(_deserialize_any(cls_map, d) for d in data)
    elif isinstance(data, (int, float, str, type(None), bool, bytes)):  # 基本类型
        return data  # 原样返回
    else:
        raise ValueError(f"Cannot deserialize type {type(data)}")  # 不认识的类型直接报错


def deserialize_type(cls_map: Dict[str, Type], data: Dict) -> Any:
    """按 __type__ 标记把字典还原为对象；Tensor 走字节级还原，其余查类表重建"""
    type_name = data["__type__"]  # 取出类型标记
    # we can only serialize 1D tensor for now  # 张量还原分支
    if type_name == "Tensor":
        buffer = data["buffer"]  # 原始字节
        dtype_str = data["dtype"].replace("torch.", "")  # 去掉前缀以便映射到 NumPy dtype
        np_dtype = getattr(np, dtype_str)  # "int32" -> np.int32
        assert isinstance(buffer, bytes)  # 确保拿到的是字节串
        np_tensor = np.frombuffer(buffer, dtype=np_dtype)  # 从字节缓冲区还原 NumPy 数组（零拷贝视图）
        return torch.from_numpy(np_tensor.copy())  # copy() 使张量持有独立内存（避免共享只读缓冲）

    cls = cls_map[type_name]  # 按类名在查找表中取类（调用方传入 globals()）
    kwargs = {}  # 构造参数字典
    for k, v in data.items():  # 遍历序列化字段
        if k == "__type__":  # 跳过类型标记本身（不是构造参数）
            continue
        kwargs[k] = _deserialize_any(cls_map, v)  # 字段值递归还原
    return cls(**kwargs)  # 用还原后的字段重建对象
