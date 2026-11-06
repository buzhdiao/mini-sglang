# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# multiprocessing：ack 回报队列的类型
import multiprocessing as mp
# 类型注解工具
from typing import List

# 推理模式装饰器（本进程也有少量 torch 操作）
import torch
# 消息协议：分词/反解码/中止请求与各类批量消息
from minisgl.message import (
    AbortBackendMsg,
    AbortMsg,
    BaseBackendMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchBackendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserMsg,
    UserReply,
)
# 同步 ZMQ 队列与 tokenizer 加载工具
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    if isinstance(msg, BatchTokenizerMsg):
        # 批量消息：直接返回内层数据列表
        return msg.data
    # 单条消息：包装成列表统一处理
    return [msg]


# tokenizer 子进程主函数：兼做分词（请求方向）与反解码（结果方向）的中间站
@torch.inference_mode()
def tokenize_worker(
    *,
    tokenizer_path: str,
    addr: str,
    create: bool,
    backend_addr: str,
    frontend_addr: str,
    local_bs: int,
    tokenizer_id: int = -1,
    model_source: str = "huggingface",
    ack_queue: mp.Queue[str] | None = None,
) -> None:
    # PUSH 到调度器：发送分词结果/中止转发（连接由调度器创建）
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)
    # PUSH 到前端：发送反解码后的用户回包（连接由前端创建）
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)
    # PULL 收包：接收前端发来的分词请求与调度器发来的 token 流
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)
    # 本进程单次合批的请求数上限必须为正
    assert local_bs > 0
    # 加载 tokenizer（词表/特殊 token）
    tokenizer = load_tokenizer(tokenizer_path)
    # 进程专属日志器（按编号区分多个 tokenizer 进程）
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")

    # 延迟导入两个管理器（避免模块加载顺序问题）
    from .detokenize import DetokenizeManager
    from .tokenize import TokenizeManager

    # 分词管理器：文本 -> token id
    tokenize_manager = TokenizeManager(tokenizer)
    # 反解码管理器：token id -> 增量文本
    detokenize_manager = DetokenizeManager(tokenizer)

    if ack_queue is not None:
        # 向主进程回报就绪（启动方要等所有子进程确认后才开始服务）
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")

    try:
        while True:
            # 阻塞取一条消息并拆包（可能是批量）
            pending_msg = _unwrap_msg(recv_listener.get())
            while len(pending_msg) < local_bs and not recv_listener.empty():
                # 凑批：队列里还有消息且未达合批上限就继续取，
                # 攒成一个大批减少后续每步的处理开销
                pending_msg.extend(_unwrap_msg(recv_listener.get()))

            logger.debug(f"Received {len(pending_msg)} messages")

            # 按类型把混合消息流分成三路
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]
            # 三路之和必须等于总数（不能有未知消息混入）
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)
            if len(detokenize_msg) > 0:
                # 反解码：token 流 -> 每条请求的增量文本
                replies = detokenize_manager.detokenize(detokenize_msg)
                batch_output = BatchFrontendMsg(
                    data=[
                        # 组装用户回包：uid + 增量文本 + 结束标志
                        UserReply(
                            uid=msg.uid,
                            incremental_output=reply,
                            finished=msg.finished,
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    # 只有一条时拆掉批包装，减少前端解包成本
                    batch_output = batch_output.data[0]
                # 发给前端（API server）
                send_frontend.put(batch_output)

            if len(tokenize_msg) > 0:
                # 分词：文本 -> token id 张量列表
                tensors = tokenize_manager.tokenize(tokenize_msg)
                batch_output = BatchBackendMsg(
                    data=[
                        # 组装用户请求：uid + 输入 token + 采样参数
                        UserMsg(
                            uid=msg.uid,
                            input_ids=t,
                            sampling_params=msg.sampling_params,
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:
                    # 单条拆包
                    batch_output = batch_output.data[0]
                # 发给调度器
                send_backend.put(batch_output)
            if len(abort_msg) > 0:
                batch_output = BatchBackendMsg(
                    # 中止消息转换成后端格式原样转发
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]
                )
                if len(batch_output.data) == 1:
                    # 单条拆包
                    batch_output = batch_output.data[0]
                # 转发给调度器
                send_backend.put(batch_output)
    except KeyboardInterrupt:
        # 收到退出信号（主进程通知或 Ctrl-C）：静默退出
        pass
