# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# 类型注解工具；Final 用于声明不可重新绑定的属性
from typing import TYPE_CHECKING, Final, List

import torch
# 消息类型：后端消息基类/批量消息、反解码结果消息等（ZMQ 上传输的协议单元）
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg
# ZMQ 收发队列封装：PULL/PUSH 用于点对点，PUB/SUB 用于一对多广播
from minisgl.utils import ZmqPubQueue, ZmqPullQueue, ZmqPushQueue, ZmqSubQueue, init_logger

# 仅类型检查时导入调度器配置，避免运行时循环依赖
if TYPE_CHECKING:
    from .config import SchedulerConfig

# 本模块日志器
logger = init_logger(__name__)


# 调度器 I/O 混入类：封装调度器与 tokenizer（反解码）进程之间的通信
class SchedulerIOMixin:
    """
    Mixin class for Scheduler I/O operations.

    This class handles the communication between the scheduler and the tokenizer.

    Public Utilities:
        receive_msg: Function to receive messages from the tokenizer.
        send_result: Function to send results back to the tokenizer.
        sync_all_ranks: Function to synchronize all ranks on CPU side.
    """

    def __init__(self, config: SchedulerConfig, tp_cpu_group: torch.distributed.ProcessGroup):
        # 取 TP 并行拓扑信息（rank、size 等）
        tp_info = config.tp_info
        # 保存 CPU 侧 TP 进程组（用于 CPU 屏障同步；Final 表示之后不可重新赋值）
        self.tp_cpu_group: Final = tp_cpu_group
        if config.offline_mode:
            # 离线模式：收发直接绑定为子类实现的进程内方法，不建任何 ZMQ 链路
            self.receive_msg = self.offline_receive_msg
            self.send_result = self.offline_send_result
            return  # early exit

        if tp_info.is_primary():
            # rank0 独占与 tokenizer 的直连：PULL 拉取用户请求（本端创建端点）
            self._recv_from_tokenizer: Final = ZmqPullQueue(
                config.zmq_backend_addr,
                create=True,
                decoder=BaseBackendMsg.decoder,
            )
            # PUSH 把反解码前的 token 消息推给 tokenizer 进程
            self._send_into_tokenizer: Final = ZmqPushQueue(
                config.zmq_detokenizer_addr,
                create=config.backend_create_detokenizer_link,
                encoder=BaseTokenizerMsg.encoder,
            )

        # 默认（TP=1）收发实现：单 rank 直连版本
        recv = self._recv_msg_single_rank
        send = self._reply_tokenizer_rank0
        if tp_info.size > 1:
            # TP>1：只有 rank0 直连 tokenizer，其余 rank 通过广播获取相同输入
            if tp_info.is_primary():
                # rank0：从 tokenizer 收到消息后转发广播给其他 rank
                recv = self._recv_msg_multi_rank0
                # PUB 套接字：把原始字节消息广播出去
                self._send_into_ranks: Final = ZmqPubQueue(
                    config.zmq_scheduler_broadcast_addr, create=True, encoder=BaseBackendMsg.encoder
                )
            else:
                # 非 rank0：从广播订阅消息，回包时静默（只 rank0 联系 tokenizer）
                recv = self._recv_msg_multi_rank1
                send = self._reply_tokenizer_rank1
                # SUB 套接字：订阅 rank0 的广播（create=False 表示连接而非绑定）
                self._recv_from_rank0: Final = ZmqSubQueue(
                    config.zmq_scheduler_broadcast_addr,
                    create=False,
                    decoder=BaseBackendMsg.decoder,
                )

        # 把选定的收发实现绑定到公开接口（不同拓扑对应不同方法）
        self.receive_msg = recv
        self.send_result = send

    def run_when_idle(self):
        # 阻塞接收前需要调度的钩子（由 Scheduler 覆盖：空闲时做完整性检查等后台任务）
        raise NotImplementedError("should be implemented")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # 离线模式收消息：由使用离线模式的子类实现
        raise NotImplementedError("should be implemented")

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        # 离线模式发结果：由使用离线模式的子类实现
        raise NotImplementedError("should be implemented")

    def sync_all_ranks(self) -> None:
        # CPU 侧全 rank 屏障：确保各 TP 进程进度对齐（如关停前同步）
        self.tp_cpu_group.barrier().wait()

    def _recv_msg_single_rank(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # 本轮收到的消息列表
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # 阻塞模式：先执行空闲期后台任务，再同步等第一条消息
            self.run_when_idle()
            # get() 阻塞直到有消息到达
            pending_msgs.append(self._recv_from_tokenizer.get())
        # 非阻塞地把队列里剩余的消息全部取走（避免积压）
        while not self._recv_from_tokenizer.empty():
            pending_msgs.append(self._recv_from_tokenizer.get())
        # 返回本批消息（可能为空列表）
        return pending_msgs

    def _recv_msg_multi_rank0(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # rank0 版本：收消息并广播，同时向其他 rank 同步消息条数
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # 阻塞时先做空闲期任务
            self.run_when_idle()
            # 取原始字节（不解码），因为要先原样广播给其他 rank
            raw = self._recv_from_tokenizer.get_raw()
            # 把原始字节放入广播队列
            self._send_into_ranks.put_raw(raw)
            # 本地解码成消息对象
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))

        # 非阻塞取走队列剩余的原始消息
        pending_raw_msgs: List[bytes] = []
        while not self._recv_from_tokenizer.empty():
            pending_raw_msgs.append(self._recv_from_tokenizer.get_raw())

        # broadcast the number of raw messages to all ranks
        # 把本轮消息条数广播给其他 rank：ZMQ PUB/SUB 是尽力而为、会丢消息的，
        # 必须用确定性的条数（经 NCCL 广播）让各 rank 精确对齐收多少条
        src_tensor = torch.tensor(len(pending_raw_msgs))
        # NCCL CPU 组广播（阻塞等待完成）
        self.tp_cpu_group.broadcast(src_tensor, root=0).wait()

        for raw in pending_raw_msgs:
            # 逐条广播原始消息并本地解码
            self._send_into_ranks.put_raw(raw)
            pending_msgs.append(self._recv_from_tokenizer.decode(raw))
        # 返回 rank0 本轮全部消息
        return pending_msgs

    def _recv_msg_multi_rank1(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # 非 rank0 版本：按 rank0 广播的条数精确接收
        pending_msgs: List[BaseBackendMsg] = []
        if blocking:
            # 阻塞时先做空闲期任务，再同步等第一条广播消息
            self.run_when_idle()
            pending_msgs.append(self._recv_from_rank0.get())

        # ensure all ranks have the same number of raw messages
        # 接收 rank0 广播的消息条数（初始 -1 便于排错）
        dst_tensor = torch.tensor(-1)
        # NCCL 广播保证所有 rank 看到同一个条数
        self.tp_cpu_group.broadcast(dst_tensor, root=0).wait()
        # 取出条数
        dst_length = int(dst_tensor.item())

        for _ in range(dst_length):
            # 精确接收对应条数的消息（多收/少收都会导致 rank 间错位）
            pending_msgs.append(self._recv_from_rank0.get())
        # 返回与 rank0 完全一致的消息列表
        return pending_msgs

    def _reply_tokenizer_rank0(self, reply: List[DetokenizeMsg]) -> None:
        # 本轮回包条数
        num_reply = len(reply)
        # 打印调试日志（仅 rank0）
        logger.debug_rank0(f"Replying to tokenizer: {num_reply} messages")
        if num_reply == 1:
            # 单条消息直接推送，免去打包/解包开销
            self._send_into_tokenizer.put(reply[0])
        elif num_reply > 1:
            # 多条打包成一个批量消息，降低 ZMQ 消息条数与序列化开销
            self._send_into_tokenizer.put(BatchTokenizerMsg(data=reply))  # type: ignore

    def _reply_tokenizer_rank1(self, reply: List[DetokenizeMsg]) -> None:
        # 非 rank0 不与 tokenizer 通信（只有 rank0 汇总输出）
        _ = reply  # do nothing for non-primary ranks
