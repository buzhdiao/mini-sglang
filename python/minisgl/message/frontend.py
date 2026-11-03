from __future__ import annotations  # 启用延迟注解求值，允许前向引用类型

from dataclasses import dataclass  # dataclass 装饰器：自动生成样板方法
from typing import Dict, List  # 类型注解工具

from .utils import deserialize_type, serialize_type  # 序列化工具：对象 <-> 可 JSON 传输的字典


@dataclass  # 数据类：前端侧消息基类
class BaseFrontendMsg:
    """前端侧消息基类：编码/解码均为静态方法（与后端基类的实例方法风格不同）"""

    @staticmethod  # 静态方法：传入消息对象进行序列化
    def encoder(msg: BaseFrontendMsg) -> Dict:
        return serialize_type(msg)  # 把消息对象转为带 __type__ 标记的字典

    @staticmethod  # 静态方法：类级反序列化入口
    def decoder(json: Dict) -> BaseFrontendMsg:
        return deserialize_type(globals(), json)  # 用本模块全局命名空间还原具体子类对象


@dataclass  # 数据类：批量包装
class BatchFrontendMsg(BaseFrontendMsg):
    data: List[BaseFrontendMsg]  # 多条前端消息打成一个包，减少通信轮次


@dataclass  # 数据类：发给用户的回复
class UserReply(BaseFrontendMsg):
    uid: int  # 对应的请求 id
    incremental_output: str  # 本步新增的增量文本（已去 token 化）
    finished: bool  # 该请求是否已生成完毕
