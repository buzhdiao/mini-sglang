from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解使用前向引用

import multiprocessing as mp  # 多进程模块，仅用于 ack_queue 的类型标注（该进程本身由 launch 拉起）
from typing import List  # 类型注解 List

import torch  # PyTorch，仅用 @torch.inference_mode() 装饰器关闭梯度以省内存/提速
from minisgl.message import (  # 导入各消息类型：tokenizer 进程是消息路由中枢，需识别/构造全部消息
    AbortBackendMsg,  # 发往调度器的中止消息（由用户 AbortMsg 转换而来）
    AbortMsg,  # 来自前端的中止请求消息
    BaseBackendMsg,  # 发往调度器的消息基类（提供 encoder）
    BaseFrontendMsg,  # 发往前端的消息基类（提供 encoder）
    BaseTokenizerMsg,  # 进入 tokenizer 的消息基类（类型标注用）
    BatchBackendMsg,  # 发往调度器的批量消息包装
    BatchFrontendMsg,  # 发往前端的批量消息包装
    BatchTokenizerMsg,  # 进入 tokenizer 的批量消息包装（ZMQ 解码目标类型）
    DetokenizeMsg,  # 调度器发来的"新 token 已生成，请解码成文本"消息
    TokenizeMsg,  # 前端发来的"请把文本编码成 input_ids"消息
    UserMsg,  # 发往调度器的用户消息：input_ids + 采样参数
    UserReply,  # 发往前端的用户回复：增量文本 + 是否结束
)
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, init_logger, load_tokenizer  # ZMQ 异步收发队列封装、日志初始化、分词器加载


def _unwrap_msg(msg: BaseTokenizerMsg) -> List[BaseTokenizerMsg]:
    """把批量消息解包成单条消息列表；单条消息则包装成单元素列表，统一后续处理逻辑"""
    if isinstance(msg, BatchTokenizerMsg):  # 批量消息：直接返回内部数据列表
        return msg.data
    return [msg]  # 单条消息：包成列表，保证调用方总能迭代处理


@torch.inference_mode()  # 推理模式上下文：禁用梯度追踪与版本计数，降低内存开销
def tokenize_worker(
    *,
    tokenizer_path: str,  # 分词器所在路径（与模型同目录）
    addr: str,  # 本进程的 ZMQ PULL/路由地址（tokenizer 或 detokenizer 专属地址）
    create: bool,  # 是否由本进程创建（bind）该 ZMQ 链路；共享模式下 detokenizer 创建、tokenizer 只 connect
    backend_addr: str,  # 发往调度器（后端）的 ZMQ PUSH 地址
    frontend_addr: str,  # 发往前端（API server）的 ZMQ PUSH 地址
    local_bs: int,  # 本地攒批大小：一次循环最多合并多少条消息，减少发送次数
    tokenizer_id: int = -1,  # tokenizer 实例编号，仅用于日志区分（detokenizer 用 num_tokenizers 作为 id）
    model_source: str = "huggingface",  # 模型来源（保留参数，当前未使用）
    ack_queue: mp.Queue[str] | None = None,  # 就绪确认队列：初始化完成后向主进程报告"已就绪"
) -> None:
    """tokenizer/detokenizer worker 进程主循环：在三类消息间路由
    - DetokenizeMsg（后端→前端）：token id 解码成文本后回给前端
    - TokenizeMsg（前端→后端）：文本编码成 input_ids 后发给调度器
    - AbortMsg（前端→后端）：中止请求转发给调度器
    """
    send_backend = ZmqPushQueue(backend_addr, create=False, encoder=BaseBackendMsg.encoder)  # PUSH 队列：连到调度器侧的 socket（由调度器 bind）
    send_frontend = ZmqPushQueue(frontend_addr, create=False, encoder=BaseFrontendMsg.encoder)  # PUSH 队列：连到前端 API server 侧的 socket
    recv_listener = ZmqPullQueue(addr, create=create, decoder=BatchTokenizerMsg.decoder)  # PULL 队列：接收上游（前端或调度器）的消息，按序解码
    assert local_bs > 0  # 攒批大小必须为正数
    tokenizer = load_tokenizer(tokenizer_path)  # 加载 HuggingFace 分词器（IO 较慢，进程启动时做一次）
    logger = init_logger(__name__, f"tokenizer_{tokenizer_id}")  # 初始化带实例编号的日志器，便于多进程日志区分

    from .detokenize import DetokenizeManager  # 延迟导入：token id → 文本的增量解码器
    from .tokenize import TokenizeManager  # 延迟导入：文本 → token id 的编码器

    tokenize_manager = TokenizeManager(tokenizer)  # 编码器实例：处理 TokenizeMsg
    detokenize_manager = DetokenizeManager(tokenizer)  # 解码器实例：处理 DetokenizeMsg（维护每请求解码状态）

    if ack_queue is not None:  # 若主进程传入了确认队列，报告本进程初始化完成
        ack_queue.put(f"Tokenize server {tokenizer_id} is ready")  # 主进程会阻塞等待收齐所有 ack 再继续

    try:
        while True:  # 常驻主循环，直到收到键盘中断
            pending_msg = _unwrap_msg(recv_listener.get())  # 阻塞取出一条消息（可能本身是批量），解包成列表
            while len(pending_msg) < local_bs and not recv_listener.empty():  # 未达攒批上限且队列还有积压时继续合并
                pending_msg.extend(_unwrap_msg(recv_listener.get()))  # 取出并解包更多消息，凑成一批处理

            logger.debug(f"Received {len(pending_msg)} messages")  # 调试日志：本批消息条数

            # 按类型把本批消息分成三组，分别路由
            detokenize_msg = [m for m in pending_msg if isinstance(m, DetokenizeMsg)]  # 需要解码的（来自调度器）
            tokenize_msg = [m for m in pending_msg if isinstance(m, TokenizeMsg)]  # 需要编码的（来自前端）
            abort_msg = [m for m in pending_msg if isinstance(m, AbortMsg)]  # 中止请求（来自前端）
            assert len(detokenize_msg) + len(tokenize_msg) + len(abort_msg) == len(pending_msg)  # 保证没有遗漏的未知消息类型
            if len(detokenize_msg) > 0:  # 处理解码消息：token id → 增量文本 → 前端
                replies = detokenize_manager.detokenize(detokenize_msg)  # 批量解码，得到每条消息的增量字符串
                batch_output = BatchFrontendMsg(  # 打包成发往前端的批量消息
                    data=[
                        UserReply(  # 每个解码结果封装为用户回复
                            uid=msg.uid,  # 请求唯一 id，前端据此匹配等待中的请求
                            incremental_output=reply,  # 本轮新增文本
                            finished=msg.finished,  # 透传结束标志
                        )
                        for msg, reply in zip(detokenize_msg, replies, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:  # 只有一条时直接发单条消息，省掉批包装开销
                    batch_output = batch_output.data[0]
                send_frontend.put(batch_output)  # 异步推送给前端 API server

            if len(tokenize_msg) > 0:  # 处理编码消息：文本 → input_ids → 调度器
                tensors = tokenize_manager.tokenize(tokenize_msg)  # 批量编码，得到每条消息的输入 id 张量
                batch_output = BatchBackendMsg(  # 打包成发往调度器的批量消息
                    data=[
                        UserMsg(  # 每条封装为用户消息
                            uid=msg.uid,  # 请求唯一 id
                            input_ids=t,  # 编码后的 token id 张量
                            sampling_params=msg.sampling_params,  # 透传采样参数
                        )
                        for msg, t in zip(tokenize_msg, tensors, strict=True)
                    ]
                )
                if len(batch_output.data) == 1:  # 单条优化：去掉批包装
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)  # 异步推送给调度器
            if len(abort_msg) > 0:  # 处理中止消息：转换成后端消息转发给调度器
                batch_output = BatchBackendMsg(  # 打包成批量中止消息
                    data=[AbortBackendMsg(uid=msg.uid) for msg in abort_msg]  # 保留 uid，调度器据此定位并中止请求
                )
                if len(batch_output.data) == 1:  # 单条优化：去掉批包装
                    batch_output = batch_output.data[0]
                send_backend.put(batch_output)  # 异步推送给调度器
    except KeyboardInterrupt:  # 进程被 Ctrl-C / 终止信号打断时静默退出
        pass  # 不需要清理动作：ZMQ socket 与分词器随进程退出自动回收
