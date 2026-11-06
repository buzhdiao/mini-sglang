# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# 泛型工具：让队列类可标注消息类型 ZmqPushQueue[Msg]
from typing import Callable, Dict, Generic, TypeVar

# msgpack：二进制序列化（比 JSON 快，适合进程间高频消息）
import msgpack
# zmq：进程间消息通道
import zmq
# zmq 的 asyncio 版本：供事件循环里非阻塞收发
import zmq.asyncio

# 消息类型参数
T = TypeVar("T")


# 同步 PUSH 队列：单向发送（一发一收与 PULL 配对，无丢消息风险）
class ZmqPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        # 每个队列独立的 ZMQ 上下文
        self.context = zmq.Context()
        # 创建 PUSH 套接字
        self.socket = self.context.socket(zmq.PUSH)
        # create=True 表示本端绑定地址（服务端），否则连接对端（客户端）
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 保存编码器：消息对象 -> 可序列化 dict
        self.encoder = encoder

    def put(self, obj: T):
        # 先编码再打成 msgpack 字节流
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        # copy=False：零拷贝直接从 Python 缓冲发送，减少内存复制
        self.socket.send(event, copy=False)

    def stop(self):
        # 关闭套接字
        self.socket.close()
        # 终结上下文（回收 ZMQ 后台线程资源）
        self.context.term()


# 异步 PUSH 队列：接口为协程，配合 asyncio 事件循环使用
class ZmqAsyncPushQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        # asyncio 版 ZMQ 上下文
        self.context = zmq.asyncio.Context()
        # PUSH 套接字
        self.socket = self.context.socket(zmq.PUSH)
        # 绑定或连接
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 保存编码器
        self.encoder = encoder

    async def put(self, obj: T):
        # 编码成字节流
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        # 异步发送：发送期间让出事件循环，不阻塞其他协程
        await self.socket.send(event, copy=False)

    def stop(self):
        # 关闭套接字
        self.socket.close()
        # 终结上下文
        self.context.term()


# 同步 PULL 队列：阻塞/非阻塞接收
class ZmqPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        # ZMQ 上下文
        self.context = zmq.Context()
        # PULL 套接字（与 PUSH 配对）
        self.socket = self.context.socket(zmq.PULL)
        # 绑定或连接
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 保存解码器：dict -> 消息对象
        self.decoder = decoder

    def get(self) -> T:
        # 阻塞接收原始字节
        event = self.socket.recv()
        # 解码成消息对象（raw=False：字节自动按字符串解码）
        return self.decoder(msgpack.unpackb(event, raw=False))

    def get_raw(self) -> bytes:
        # 取原始字节不解码：TP 广播场景要原样转发给其他 rank
        return self.socket.recv()

    def decode(self, raw: bytes) -> T:
        # 把已有字节解码成消息对象（广播转发后本地自己也要用）
        return self.decoder(msgpack.unpackb(raw, raw=False))

    def empty(self) -> bool:
        # 0 超时轮询：无消息返回 True（用于非阻塞排空队列）
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        # 关闭套接字
        self.socket.close()
        # 终结上下文
        self.context.term()


# 异步 PULL 队列：await 等待消息
class ZmqAsyncPullQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        # asyncio 版上下文
        self.context = zmq.asyncio.Context()
        # PULL 套接字
        self.socket = self.context.socket(zmq.PULL)
        # 绑定或连接
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 保存解码器
        self.decoder = decoder

    async def get(self) -> T:
        # 异步阻塞等一条消息
        event = await self.socket.recv()
        # 解码成消息对象
        return self.decoder(msgpack.unpackb(event, raw=False))

    def stop(self):
        # 关闭套接字
        self.socket.close()
        # 终结上下文
        self.context.term()


# PUB 队列：一对多广播（TP>1 时 rank0 -> 其余 rank）
# 注意 PUB/SUB 会静默丢消息（无流控），只适合配合外部同步使用
class ZmqPubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        encoder: Callable[[T], Dict],
    ):
        # ZMQ 上下文
        self.context = zmq.Context()
        # PUB 套接字
        self.socket = self.context.socket(zmq.PUB)
        # 绑定或连接
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 保存编码器
        self.encoder = encoder

    def put_raw(self, raw: bytes):
        # 直接广播原始字节（rank0 收到什么就转发什么，省一次解码-再编码）
        self.socket.send(raw, copy=False)

    def put(self, obj: T):
        # 编码后广播
        event = msgpack.packb(self.encoder(obj), use_bin_type=True)
        self.socket.send(event, copy=False)

    def stop(self):
        # 关闭套接字
        self.socket.close()
        # 终结上下文
        self.context.term()


# SUB 队列：PUB 的订阅端
class ZmqSubQueue(Generic[T]):
    def __init__(
        self,
        addr: str,
        create: bool,
        decoder: Callable[[Dict], T],
    ):
        # ZMQ 上下文
        self.context = zmq.Context()
        # SUB 套接字
        self.socket = self.context.socket(zmq.SUB)
        # 绑定或连接
        self.socket.bind(addr) if create else self.socket.connect(addr)
        # 订阅所有消息（空过滤器 = 不做主题过滤）
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # 保存解码器
        self.decoder = decoder

    def get(self) -> T:
        # 阻塞接收广播消息并解码
        event = self.socket.recv()
        return self.decoder(msgpack.unpackb(event, raw=False))

    def empty(self) -> bool:
        # 0 超时轮询判空
        return self.socket.poll(timeout=0) == 0

    def stop(self):
        # 关闭套接字
        self.socket.close()
        # 终结上下文
        self.context.term()
