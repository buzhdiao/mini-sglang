from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from typing import TYPE_CHECKING, Final, List  # 类型注解工具：TYPE_CHECKING、Final（不可重绑定常量）、List

import torch  # PyTorch：构造 CPU 张量用于跨 rank 广播消息计数
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg  # 消息类型：后端消息基类、分词器消息基类、批量消息、去 token 化结果
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger  # ZMQ 队列封装（PUB/PULL/PUSH/SUB）与日志工具

if TYPE_CHECKING:  # 仅静态类型检查时导入，避免运行时循环依赖
    from .config import SchedulerConfig  # 调度器配置类型

logger = init_logger(__name__)  # 模块级日志器


class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """
    # 调度器 I/O 混入类：封装调度器与 tokenizer 进程及各 TP rank 之间的 ZMQ 通信。
    # 根据离线模式与 TP 规模，在 __init__ 中动态绑定 receive_msg / send_result 的具体实现。

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        # 初始化 I/O：按 TP 拓扑选择收发路径（单 rank 直收 / rank0 收并广播 / 非 rank0 订阅）
        tp_info = config.tp_info  # 张量并行拓扑信息（rank、size、is_primary）
        self.tp_cpu_group: Final = tp_cpu_group  # CPU 侧进程组：用于 barrier 同步与消息计数广播
        if config.offline_mode:  # 离线模式：不建 ZMQ，收发由子类覆写绑定
            self.receive_msg = self.offline_receive_msg  # 绑定离线接收占位实现
            self.send_result = self.offline_send_result  # 绑定离线发送占位实现
            return  # early exit  # 提前返回，跳过 ZMQ 初始化

        if tp_info.is_primary():  # 主 rank（rank0）负责创建（bind）ZMQ 端点
            self._recv_from_tokenizer: Final = ZmqPullQueue(  # PULL 队列：从 tokenizer 接收请求消息
                config.zmq_backend_addr,  # 绑定地址（ipc:///tmp/minisgl_0.pid=x）
                create=True,  # 本端创建（bind）
                decoder=BaseBackendMsg.decoder,  # 字节流 -> 消息对象的解码器
            )
            self._send_into_tokenizer: Final = ZmqPushQueue(  # PUSH 队列：向 detokenizer 发送结果
                config.zmq_detokenizer_addr,  # 目标地址（ipc:///tmp/minisgl_1.pid=x）
                create=config.backend_create_detokenizer_link,  # 是否由本端创建连接
                encoder=BaseTokenizerMsg.encoder,  # 消息对象 -> 字节流的编码器
            )

        recv = self._recv_msg_single_rank  # 默认接收实现：单 rank 直收
        send = self._reply_tokenizer_rank0  # 默认发送实现：rank0 直接回复
        if tp_info.size > 1:  # TP>1：需在 rank 间分发请求、汇聚结果
            if tp_info.is_primary():  # rank0：接收后向其他 rank 广播
                recv = self._recv_msg_multi_rank0  # 切换为 rank0 多卡接收实现
                self._send_into_ranks: Final = ZmqPubQueue(  # PUB 队列：向其他 rank 广播请求消息
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:  # 非 rank0：订阅 rank0 广播，回复操作为空
                recv = self._recv_msg_multi_rank1  # 切换为非 rank0 接收实现
                send = self._reply_tokenizer_rank1  # 切换为空回复实现（仅 rank0 对外）
                self._recv_from_rank0: Final = ZmqSubQueue(  # SUB 队列：订阅 rank0 的广播
                    config.zmq_scheduler_broadcast_addr,
                    create=False,  # 非 bind 端（connect 到 rank0）
                    decoder=BaseBackendMsg.decoder,
                )

        self.receive_msg = recv  # 绑定最终选定的接收实现
        self.send_result = send  # 绑定最终选定的发送实现

    def run_when_idle(self):
        # 空闲钩子：阻塞等待首条消息前调用，供子类做后台维护（Scheduler 中做缓存一致性检查）
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # 离线模式接收消息的占位实现：由离线驱动子类覆写
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        # 离线模式发送结果的占位实现：由离线驱动子类覆写
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        # CPU 侧全 rank 屏障：确保各 TP rank 进度对齐（如关闭引擎前）
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # 单 rank 接收：blocking 时先阻塞等首条消息，再非阻塞取空队列（一次尽量多处理）
        pending_msgs: List[BaseBackendMsg] = []  # 本轮收到的消息列表
        if blocking:  # 阻塞模式：至少等到一条消息
            self.run_when_idle()  # 空闲钩子（如做一致性检查）
            pending_msgs.append(self._recv_from_tokenizer.get())  # 阻塞取首条
        while not self._recv_from_tokenizer.empty():  # 非阻塞地取空队列剩余消息
            pending_msgs.append(self._recv_from_tokenizer.get())
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # TP>1 时 rank0 的接收：收到的原始消息转发广播给其他 rank，并返回解码后的消息
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:  # 阻塞模式：先等到首条
            self.run_when_idle()  # 空闲钩子
            raw = self._recv_from_tokenizer.get_raw()  # 取原始字节（便于原样转发）
            self._send_into_ranks.put_raw(raw)  # 转发给其他 rank
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))  # 本地解码入列表

        pending_raw_msgs: List[bytes] = []  # 队列中剩余的原始消息
        while not self._recv_from_tokenizer.empty():  # 非阻塞取空队列
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks  # 广播消息条数：让其他 rank 知道要收几条
        src_tensor = torch.tensor(len(pending_raw_msgs))  # rank0 的计数张量
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()  # 广播并等待完成

        for raw in pending_raw_msgs:  # 逐条转发给其他 rank 并本地解码
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # TP>1 时非 rank0 的接收：先同步条数，再从 rank0 广播中逐条取出
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:  # 阻塞模式：先等一条（保证 idle 钩子有执行机会）
            self.run_when_idle()  # 空闲钩子
            pending_msgs.append(self._recv_from_rank0.get())  # 阻塞取首条广播消息

        # ensure all ranks have the same number of raw messages  # 与 rank0 对齐消息条数
        dst_tensor = torch.tensor(-1)  # 接收缓冲：-1 为占位初值
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()  # 接收 rank0 广播的计数
        dst_length = int(dst_tensor.item())  # 需要接收的消息条数

        for _ in range(dst_length):  # 按条数逐条取出（与 rank0 转发一一对应）
            pending_msgs.append(self._recv_from_rank0.get())
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        # rank0 回复 tokenizer：单条直接发送，多条打包为批量消息（减少 ZMQ 发送次数）
        num_reply = len(reply)  # 回复条数
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")  # 调试日志（仅 rank0 记录）
        if num_reply == 1:  # 单条：直接发送
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:  # 多条：打包成批量消息一次发送
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore
        # num_reply == 0：无结果时不发送

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        _ = reply  # do nothing for non-primary ranks  # 非 rank0 不回复 tokenizer（对外输出只经 rank0）
