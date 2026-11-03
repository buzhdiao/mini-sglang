from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from typing import Callable, Dict, Generic, TypeVar  # 泛型工具：让队列对消息类型 T 做类型标注

import msgpack  # 高效二进制序列化库，用于跨进程消息的编码/解码
import zmq  # ZeroMQ 同步套接字库
import zmq.asyncio  # ZeroMQ 的 asyncio 异步版本（提供异步 Context）

T = TypeVar("T")  # 消息类型占位符


class ZmqPushQueue(Generic[T]):
    """ZMQ PUSH 推送队列（同步版）：生产者一端，把对象序列化后推给下游 PULL 端（负载均衡分发）"""

    def __init__(
        self,
        addr: str,  # ZMQ 端点地址（如 ipc:///tmp/xxx.sock 或 tcp://host:port）
        create: bool,  # True=bind 监听端（服务方先启动），False=connect 连接端（客户端后接入）
        encoder: Callable[[T], Dict],  # 自定义编码器：把消息对象转成可被 msgpack 序列化的 dict
    ):
        self.context = zmq.Context()  # 创建 ZMQ 上下文（同步版，管理底层 IO 线程）
        self.socket = self.context.socket(zmq.PUSH)  # 创建 PUSH 类型套接字（推送端）
        self.socket.bind(addr) if create else self.socket.connect(addr)  # 按角色二选一：服务方 bind，客户端 connect
        self.encoder = encoder  # 保存编码器，供 put 时序列化使用

    def put(self, obj: T):
        """序列化并推送一条消息（阻塞直到消息进入发送队列）"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)  # 先编码成 dict，再序列化为二进制（bin 类型保真，str->bin）
        self.socket.send(event, copy=False)  # 零拷贝发送，减少一次内存复制

    def stop(self):
        """关闭套接字并终结上下文，释放底层资源"""
        self.socket.close()  # 先关套接字
        self.context.term()  # 再终结上下文（等待 IO 线程退出）


class ZmqAsyncPushQueue(Generic[T]):
    """ZMQ PUSH 推送队列（异步版）：接口与同步版一致，put 变为协程，可融入 asyncio 事件循环"""

    def __init__(
        self,
        addr: str,  # ZMQ 端点地址
        create: bool,  # True=bind 监听端，False=connect 连接端
        encoder: Callable[[T], Dict],  # 消息对象 -> dict 的编码器
    ):
        self.context = zmq.asyncio.Context()  # 创建异步 ZMQ 上下文（关键：必须用 asyncio 版，否则 send 不可 await）
        self.socket = self.context.socket(zmq.PUSH)  # 创建 PUSH 套接字
        self.socket.bind(addr) if create else self.socket.connect(addr)  # 服务方 bind / 客户端 connect
        self.encoder = encoder  # 保存编码器

    async def put(self, obj: T):
        """异步序列化并推送一条消息（await 使出队逻辑不阻塞事件循环）"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)  # 编码 + msgpack 序列化
        await self.socket.send(event, copy=False)  # 异步零拷贝发送

    def stop(self):
        """关闭套接字并终结上下文"""
        self.socket.close()
        self.context.term()


class ZmqPullQueue(Generic[T]):
    """ZMQ PULL 拉取队列（同步版）：消费者一端，从上游 PUSH 端阻塞接收消息"""

    def __init__(
        self,
        addr: str,  # ZMQ 端点地址
        create: bool,  # True=bind 监听端，False=connect 连接端
        decoder: Callable[[Dict], T],  # 自定义解码器：把 msgpack 解出的 dict 还原成消息对象
    ):
        self.context = zmq.Context()  # 同步 ZMQ 上下文
        self.socket = self.context.socket(zmq.PULL)  # 创建 PULL 类型套接字（拉取端）
        self.socket.bind(addr) if create else self.socket.connect(addr)  # 服务方 bind / 客户端 connect
        self.decoder = decoder  # 保存解码器

    def get(self) -> T:
        """阻塞接收并解码一条消息"""
        event = self.socket.recv()  # 阻塞等待二进制消息到达
        return self.decoder(msgpack.unpackb(event, raw=False))  # msgpack 解码（raw=False 保证 str 而非 bytes）再转业务对象

    def get_raw(self) -> bytes:
        """阻塞接收原始二进制消息（不解码，交给调用方处理）"""
        return self.socket.recv()  # 直接返回字节串

    def decode(self, raw: bytes) -> T:
        """把原始字节解码为业务对象（常与 get_raw 搭配：多路复用线程收包、单线程解码）"""
        return self.decoder(msgpack.unpackb(raw, raw=False))  # msgpack 解码 + 业务还原

    def empty(self) -> bool:
        """非阻塞探测当前是否有待收消息（True=队列空，可用于轮询调度而不阻塞）"""
        return self.socket.poll(timeout=0) == 0  # 0 超时立即返回：无事件即视为空

    def stop(self):
        """关闭套接字并终结上下文"""
        self.socket.close()
        self.context.term()


class ZmqAsyncPullQueue(Generic[T]):
    """ZMQ PULL 拉取队列（异步版）：get 变为协程，接收时不阻塞事件循环"""

    def __init__(
        self,
        addr: str,  # ZMQ 端点地址
        create: bool,  # True=bind 监听端，False=connect 连接端
        decoder: Callable[[Dict], T],  # dict -> 业务对象的解码器
    ):
        self.context = zmq.asyncio.Context()  # 异步 ZMQ 上下文
        self.socket = self.context.socket(zmq.PULL)  # PULL 套接字
        self.socket.bind(addr) if create else self.socket.connect(addr)  # 服务方 bind / 客户端 connect
        self.decoder = decoder  # 保存解码器

    async def get(self) -> T:
        """异步阻塞接收并解码一条消息（await 期间事件循环可处理其他任务）"""
        event = await self.socket.recv()  # 异步等待消息
        return self.decoder(msgpack.unpackb(event, raw=False))  # 解码并还原业务对象

    def stop(self):
        """关闭套接字并终结上下文"""
        self.socket.close()
        self.context.term()


class ZmqPubQueue(Generic[T]):
    """ZMQ PUB 发布队列：广播端，一条消息发给所有已连接的订阅者（无订阅者则消息直接丢弃）"""

    def __init__(
        self,
        addr: str,  # ZMQ 端点地址
        create: bool,  # True=bind 监听端，False=connect 连接端
        encoder: Callable[[T], Dict],  # 业务对象 -> dict 的编码器
    ):
        self.context = zmq.Context()  # 同步 ZMQ 上下文
        self.socket = self.context.socket(zmq.PUB)  # 创建 PUB 类型套接字（发布端）
        self.socket.bind(addr) if create else self.socket.connect(addr)  # 服务方 bind / 客户端 connect
        self.encoder = encoder  # 保存编码器

    def put_raw(self, raw: bytes):
        """直接广播原始字节（跳过编码，适合转发已有二进制数据）"""
        self.socket.send(raw, copy=False)  # 零拷贝发送

    def put(self, obj: T):
        """编码后广播一条消息（发送非阻塞：PUB 不排队等慢消费者，可能丢消息换取低延迟）"""
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)  # 编码 + 序列化
        self.socket.send(event, copy=False)  # 广播给所有订阅者

    def stop(self):
        """关闭套接字并终结上下文"""
        self.socket.close()
        self.context.term()


class ZmqSubQueue(Generic[T]):
    """ZMQ SUB 订阅队列：广播接收端，与 PUB 端配套使用"""

    def __init__(
        self,
        addr: str,  # ZMQ 端点地址
        create: bool,  # True=bind 监听端，False=connect 连接端
        decoder: Callable[[Dict], T],  # dict -> 业务对象的解码器
    ):
        self.context = zmq.Context()  # 同步 ZMQ 上下文
        self.socket = self.context.socket(zmq.SUB)  # 创建 SUB 类型套接字（订阅端）
        self.socket.bind(addr) if create else self.socket.connect(addr)  # 服务方 bind / 客户端 connect
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")  # 订阅空主题前缀 = 接收所有消息（SUB 默认不收任何消息，必须设置）
        self.decoder = decoder  # 保存解码器

    def get(self) -> T:
        """阻塞接收并解码一条广播消息"""
        event = self.socket.recv()  # 阻塞等待消息
        return self.decoder(msgpack.unpackb(event, raw=False))  # 解码并还原业务对象

    def empty(self) -> bool:
        """非阻塞探测是否有待收消息（True=空，用于轮询而不阻塞）"""
        return self.socket.poll(timeout=0) == 0  # 0 超时立即返回

    def stop(self):
        """关闭套接字并终结上下文"""
        self.socket.close()
        self.context.term()
