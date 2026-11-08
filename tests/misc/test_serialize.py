from __future__ import annotations  # 启用 PEP 563 延迟注解求值
from dataclasses import dataclass  # 导入 dataclass 装饰器
from typing import List  # 导入 List 类型注解

from minisgl.core import SamplingParams  # 导入采样参数（真实业务类型）
import torch  # 导入 torch，测试张量字段的序列化
from minisgl.message import BatchBackendMsg, UserMsg  # 导入真实消息类型：backend 批消息与用户请求消息
from minisgl.message.utils import serialize_type, deserialize_type  # 导入通用序列化/反序列化工具
from minisgl.utils import call_if_main, init_logger  # 导入"仅主模块执行"装饰器与日志工具

logger = init_logger(__name__)  # 创建本模块 logger


@dataclass  # 数据类装饰器
class A:
    """测试用递归数据类：含 int、str、嵌套自身列表、torch 张量四种字段，覆盖各类序列化路径。"""

    x: int  # 整数字段
    y: str  # 字符串字段
    z: List[A]  # 嵌套自身类型的列表（测试递归序列化）
    w: torch.Tensor  # torch 张量字段（测试二进制张量序列化）


@call_if_main()  # 装饰器：仅本文件作为主模块运行时执行
def test_serialize_deserialize():
    """序列化冒烟测试：自定义递归类型与真实消息类型分别做 序列化→反序列化 往返并打印结果。"""

    t = torch.tensor([1, 2, 3], dtype=torch.int32)  # 测试用张量
    x = A(10, "hello", [A(20, "world", [], t)], t)  # 构造含嵌套对象与张量的实例
    data = serialize_type(x)  # 序列化为字节串（自动处理递归结构与张量）
    logger.info(data)  # 打印序列化结果
    y = deserialize_type({"A": A}, data)  # 反序列化：需提供 类型名→类型 的映射表以还原自定义类
    logger.info(y)  # 打印还原结果（人工比对一致性）

    u = BatchBackendMsg([UserMsg(uid=0, input_ids=t, sampling_params=SamplingParams())])  # 构造真实业务消息：含一条用户请求
    result = u.decoder(u.encoder())  # 用消息自带的编码器/解码器做往返（进程间通信的实际路径）
    logger.info(u)  # 打印原消息
    logger.info(result)  # 打印往返后的消息（人工比对字段是否无损还原）
