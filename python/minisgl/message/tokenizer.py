from __future__ import annotations  # 启用延迟注解求值，允许前向引用类型

from dataclasses import dataclass  # dataclass 装饰器：自动生成样板方法
from typing import Dict, List  # 类型注解工具

from minisgl.core import SamplingParams  # 采样参数：随编码请求一起传递

from .utils import deserialize_type, serialize_type  # 序列化工具：对象 <-> 可 JSON 传输的字典


@dataclass  # 数据类：分词器侧消息基类
class BaseTokenizerMsg:
    """分词器侧消息基类：提供静态的序列化/反序列化入口"""

    @staticmethod  # 静态方法：传入消息对象进行序列化
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        return serialize_type(msg)  # 把消息对象转为带 __type__ 标记的字典

    @staticmethod  # 静态方法：类级反序列化入口
    def decoder(json: Dict) -> BaseTokenizerMsg:
        return deserialize_type(globals(), json)  # 用本模块全局命名空间还原具体子类对象


@dataclass  # 数据类：批量包装
class BatchTokenizerMsg(BaseTokenizerMsg):
    data: List[BaseTokenizerMsg]  # 多条分词器消息打成一个包，减少通信轮次


@dataclass  # 数据类：解码（去 token 化）结果消息
class DetokenizeMsg(BaseTokenizerMsg):
    uid: int  # 对应的请求 id
    next_token: int  # 本步新生成的 token id
    finished: bool  # 该请求是否已生成完毕


@dataclass  # 数据类：编码（分词）请求消息
class TokenizeMsg(BaseTokenizerMsg):
    uid: int  # 请求唯一 id
    text: str | List[Dict[str, str]]  # 原始文本，或 chat 模板用的对话消息列表
    sampling_params: SamplingParams  # 该请求的采样参数


@dataclass  # 数据类：中止请求消息
class AbortMsg(BaseTokenizerMsg):
    uid: int  # 要中止的请求 id
