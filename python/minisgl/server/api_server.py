# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# asyncio：异步事件（等待 ack）与协程任务
import asyncio
# json：流式 chunk 的序列化
import json
# time：生成模型卡时间戳等
import time
# asynccontextmanager：定义 FastAPI 的 lifespan 生命周期钩子
from contextlib import asynccontextmanager
# dataclass/field：生成前端管理器类
from dataclasses import dataclass, field
# 类型注解工具
from typing import Callable, Dict, List, Literal, Tuple

# uvicorn：ASGI 服务器，承载 FastAPI 应用
import uvicorn
# FastAPI/Request：Web 框架路由与请求对象
from fastapi import FastAPI, Request
# StreamingResponse：以 SSE 流式返回生成结果
from fastapi.responses import StreamingResponse
# 采样参数（请求 -> tokenizer -> 调度器 一路携带）
from minisgl.core import SamplingParams
# 环境变量（shell 模式的采样默认值等）
from minisgl.env import ENV
# 消息协议：中止/批量/分词请求与用户回包
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BaseTokenizerMsg,
    BatchFrontendMsg,
    TokenizeMsg,
    UserReply,
)
# 异步 ZMQ 队列：与 tokenizer 进程收发消息
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger
# prompt_toolkit：交互式终端（补全、异步输入）
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
# pydantic：请求体校验模型
from pydantic import BaseModel, Field
# starlette 后台任务：响应结束后执行（用于中止清理）
from starlette.background import BackgroundTask

# 服务器参数（含 ZMQ 地址等）
from .args import ServerArgs

# 前端 API 日志器（固定名称前缀 FrontendAPI）
logger = init_logger(__name__, "FrontendAPI")

# 全局前端状态：单例，由 run_api_server 初始化
_GLOBAL_STATE = None


def get_global_state() -> FrontendManager:
    # 引用模块级单例
    global _GLOBAL_STATE
    # 必须先初始化才能使用
    assert _GLOBAL_STATE is not None, "Global state is not initialized"
    # 返回全局前端管理器
    return _GLOBAL_STATE


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    if isinstance(msg, BatchFrontendMsg):
        # 批量回包：拆开成单条列表
        result = []
        for reply in msg.data:
            # 逐条校验类型
            assert isinstance(reply, UserReply)
            # 收集
            result.append(reply)
        # 返回拆包结果
        return result
    # 单条回包：校验类型
    assert isinstance(msg, UserReply)
    # 包装成单元素列表，统一下游处理
    return [msg]


# /generate 端点的简单请求体
class GenerateRequest(BaseModel):
    # 输入提示文本
    prompt: str
    # 最多生成的 token 数
    max_tokens: int
    # 是否忽略 eos（生成到 max_tokens 为止）
    ignore_eos: bool = False


# OpenAI 风格的聊天消息
class Message(BaseModel):
    # 角色：system/user/assistant 三选一
    role: Literal["system", "user", "assistant"]
    # 消息内容
    content: str


# OpenAI 兼容端点的统一请求体（completions 与 chat-completions 共用）
class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""

    # 模型名（当前仅用于回显）
    model: str

    # 补续式输入（与 messages 二选一）
    prompt: str | None = None
    # 聊天式消息列表（与 prompt 二选一）
    messages: List[Message] | None = None

    # 最大生成 token 数（OpenAI 默认 16）
    max_tokens: int = 16
    # 采样温度：越高越随机
    temperature: float = 1.0

    # top-k 采样（-1 表示不启用）
    top_k: int = -1
    # top-p（核）采样
    top_p: float = 1.0
    # 每个请求生成的候选数（本实现只支持 1）
    n: int = 1
    # 是否流式返回
    stream: bool = False
    # 停止字符串（预留）
    stop: List[str] = []
    # 存在惩罚：抑制重复出现过的 token
    presence_penalty: float = 0.0
    # 频率惩罚：按出现次数抑制重复
    frequency_penalty: float = 0.0

    # 是否忽略 eos（透传给调度器）
    ignore_eos: bool = False


# OpenAI 风格的模型卡
class ModelCard(BaseModel):
    # 模型 id（这里用模型路径）
    id: str
    # OpenAI 协议固定字段
    object: str = "model"
    # 创建时间戳（每次请求时生成）
    created: int = Field(default_factory=lambda: int(time.time()))
    # 归属方标识
    owned_by: str = "mini-sglang"
    # 根模型名
    root: str


# OpenAI 风格的模型列表
class ModelList(BaseModel):
    # 协议固定字段
    object: str = "list"
    # 模型卡数组
    data: List[ModelCard] = Field(default_factory=list)


# 前端管理器：维护 uid、ack 通道与事件，衔接 FastAPI 与 tokenizer 进程
@dataclass
class FrontendManager:
    # 服务器参数
    config: ServerArgs
    # 异步 PUSH 队列：把请求发给 tokenizer
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]
    # 异步 PULL 队列：接收 tokenizer 的回包
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]
    # uid 发号器（自增）
    uid_counter: int = 0
    # 监听协程是否已启动（只启动一次）
    initialized: bool = False
    # uid -> 已收到未消费的回包列表
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)
    # uid -> 事件对象（收到新回包时 set，唤醒等待的协程）
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)

    def new_user(self) -> int:
        # 取当前计数作为新 uid
        uid = self.uid_counter
        # 计数自增
        self.uid_counter += 1
        # 为该 uid 建立空的 ack 收件箱
        self.ack_map[uid] = []
        # 为该 uid 建立未触发的等待事件
        self.event_map[uid] = asyncio.Event()
        # 返回新 uid
        return uid

    async def listen(self):
        # 后台监听协程：持续收 tokenizer 回包并分发
        while True:
            # 异步等待一条消息（可能是批量）
            msg = await self.recv_tokenizer.get()
            for msg in _unwrap_msg(msg):
                # 拆包后逐条分发
                if msg.uid not in self.ack_map:
                    # 该 uid 已不存在（请求被中止/清理）：丢弃迟到的回包
                    continue
                # 追加到该 uid 的收件箱
                self.ack_map[msg.uid].append(msg)
                # set 事件唤醒正在等待的请求协程
                self.event_map[msg.uid].set()

    def _create_listener_once(self):
        if not self.initialized:
            # 惰性启动后台监听任务（首次发请求时才需要）
            asyncio.create_task(self.listen())
            # 标记已启动，保证只启动一次
            self.initialized = True

    async def send_one(self, msg: BaseTokenizerMsg):
        # 确保监听协程在运行
        self._create_listener_once()
        # 异步把消息推给 tokenizer 进程
        await self.send_tokenizer.put(msg)

    async def wait_for_ack(self, uid: int):
        # 取该 uid 的等待事件
        event = self.event_map[uid]

        while True:
            # 挂起等待新回包到达
            await event.wait()
            # 立即清事件，下一轮继续等新增量
            event.clear()

            # 取走收件箱中全部未消费回包
            pending = self.ack_map[uid]
            # 清空收件箱（换新列表，期间新回包再入新列表）
            self.ack_map[uid] = []
            # 记录最后一条回包（用于判断结束）
            ack = None
            for ack in pending:
                # 以生成器形式逐条产出（流式协议的基础）
                yield ack
            if ack and ack.finished:
                # 最后一条带结束标志：终止等待循环
                break

        # 请求结束：清理该 uid 的全部状态
        del self.ack_map[uid]
        del self.event_map[uid]

    async def stream_generate(self, uid: int):
        # SSE 流：逐条产出增量文本
        async for ack in self.wait_for_ack(uid):
            # 产出 data: <增量文本>\n 格式的 SSE 帧
            yield f"data: {ack.incremental_output}\n".encode()
            if ack.finished:
                # 结束：停止产出
                break
        # SSE 约定的结束标记
        yield "data: [DONE]\n".encode()
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_chat_completions(self, uid: int):
        # 是否为第一个 chunk（需要先带 role）
        first_chunk = True
        async for ack in self.wait_for_ack(uid):
            # 本 chunk 的增量字段
            delta = {}
            if first_chunk:
                # 首个 chunk 携带角色声明
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:
                # 有增量文本才填 content（避免空 chunk）
                delta["content"] = ack.incremental_output

            # 组装 OpenAI chat.completion.chunk 结构
            chunk = {
                "id": f"cmpl-{uid}",
                "object": "text_completion.chunk",
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],
            }
            # SSE 帧以空行结尾（\n\n）
            yield f"data: {json.dumps(chunk)}\n\n".encode()

            if ack.finished:
                # 生成结束
                break

        # send final finish_reason
        # 结束 chunk：delta 为空、finish_reason=stop
        end_chunk = {
            "id": f"cmpl-{uid}",
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
        # 产出结束 chunk
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()
        # SSE 结束标记
        yield b"data: [DONE]\n\n"
        logger.debug("Finished streaming response for user %s", uid)

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        try:
            async for chunk in generator:
                # detect if the client has disconnected
                # 每产出一个 chunk 前检查客户端是否已断开
                if await request.is_disconnected():
                    # 断开则记日志
                    logger.info("Client disconnected for user %s", uid)
                    # 抛取消异常终止流
                    raise asyncio.CancelledError
                # 正常透传 chunk
                yield chunk
        except asyncio.CancelledError:
            # 客户端断开/协程被取消：异步通知调度器中止该请求（不阻塞取消流程）
            asyncio.create_task(self.abort_user(uid))
            # 继续向上传播取消
            raise

    async def abort_user(self, uid: int):
        # 稍作等待：给正在传播的流取消留一点时间，避免与正常收尾竞争
        await asyncio.sleep(0.1)
        if uid in self.ack_map:
            # 清理收件箱，后续迟到回包会被 listen 丢弃
            del self.ack_map[uid]
        if uid in self.event_map:
            # 清理等待事件
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)
        # 发送中止消息给 tokenizer -> 调度器
        await self.send_one(AbortMsg(uid=uid))

    def shutdown(self):
        # 停止发送队列（关闭 ZMQ 端点与后台线程）
        self.send_tokenizer.stop()
        # 停止接收队列
        self.recv_tokenizer.stop()


# FastAPI lifespan：yield 前是启动逻辑（此处为空），yield 后是关停逻辑
@asynccontextmanager
async def lifespan(_: FastAPI):
    # 应用运行期
    yield
    # shutdown code here
    # 应用关停：清理全局状态（关闭 ZMQ）
    global _GLOBAL_STATE
    if _GLOBAL_STATE is not None:
        _GLOBAL_STATE.shutdown()


# FastAPI 应用实例（挂载 lifespan）
app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    # 记录请求（调试级）
    logger.debug("Received generate request %s", req)
    # 取全局前端状态
    state = get_global_state()
    # 为该请求分配 uid 并注册 ack 通道
    uid = state.new_user()
    await state.send_one(
        # 发送分词请求：文本 + 采样参数
        TokenizeMsg(
            uid=uid,
            text=req.prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
            ),
        )
    )

    return StreamingResponse(
        # 返回 SSE 流：带客户端断开检测的包装生成器
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),
        media_type="text/event-stream",
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])
async def v1_root():
    # /v1 根路径健康检查端点
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    # 取全局前端状态
    state = get_global_state()
    if req.messages:
        # 聊天式：把消息列表转成 dict 列表（tokenizer 端会套 chat template）
        prompt = [msg.model_dump() for msg in req.messages]
    else:
        # 补续式：必须提供 prompt
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"
        prompt = req.prompt

    # TODO: support more sampling parameters
    # 分配 uid 并注册 ack 通道
    uid = state.new_user()
    await state.send_one(
        # 发送分词请求（携带完整采样参数）
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    if req.stream:
        # 流式：返回 OpenAI chunk 格式的 SSE 流
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),
            media_type="text/event-stream",
        )

    # Non-streaming: collect all chunks and return a single JSON response
    # 非流式：攒齐所有增量后一次性返回完整 JSON
    full_content = ""
    async for ack in state.wait_for_ack(uid):
        # 累加每段增量文本
        full_content += ack.incremental_output
        if ack.finished:
            # 结束即停
            break

    # 组装 OpenAI chat.completion 响应
    return {
        "id": f"chatcmpl-{uid}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": full_content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            # 用量统计未实现，占位 0
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")
async def available_models():
    # 取全局状态
    state = get_global_state()
    # 返回单条模型卡（以模型路径作为 id/root）
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])


async def shell_completion(req: OpenAICompletionRequest):
    # 取全局状态
    state = get_global_state()
    # shell 模式只支持聊天式输入
    assert req.messages is not None, "Shell completion only supports chat-completions"
    # 消息转 dict 列表
    prompt = [msg.model_dump() for msg in req.messages]

    # TODO: support more sampling parameters
    # 分配 uid
    uid = state.new_user()
    await state.send_one(
        # 发送分词请求
        TokenizeMsg(
            uid=uid,
            text=prompt,
            sampling_params=SamplingParams(
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    async def _abort():
        # 请求结束后的清理：中止该 uid（防止残留在调度器里）
        await state.abort_user(uid)

    return StreamingResponse(
        # 简单 SSE 流（shell 内部消费，不做断开检测）
        state.stream_generate(uid),
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),
    )


# 内置交互式终端：与本地推理服务对话（学习/调试用）

async def shell():
    # 支持的斜杠命令（用于补全提示）
    commands = ["/exit", "/reset"]
    # 单词补全器
    completer = WordCompleter(commands)
    # 异步输入会话（提示符 "$ "）
    session = PromptSession("$ ", completer=completer)

    try:
        # 对话历史：(用户输入, 助手回复) 对列表
        history: List[Tuple[str, str]] = []
        while True:
            # 异步读取一行输入并去除首尾空白
            cmd = (await session.prompt_async()).strip()
            if cmd == "":
                # 空行：重新提示
                continue
            if cmd.startswith("/"):
                # 斜杠命令分支
                if cmd == "/exit":
                    # 退出 shell（返回后走 finally 清理）
                    return
                if cmd == "/reset":
                    # 清空对话历史
                    history = []
                    continue
                # 未知命令直接报错
                raise ValueError(f"Unknown command: {cmd}")
            # 把历史轮次展开成消息列表（user/assistant 交替）
            history_messages: List[Message] = []
            for user_msg, assistant_msg in history:
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            # 组装本轮请求：历史 + 当前输入；采样参数来自环境变量
            req = OpenAICompletionRequest(
                model="",
                messages=history_messages + [Message(role="user", content=cmd)],
                max_tokens=ENV.SHELL_MAX_TOKENS.value,
                top_k=ENV.SHELL_TOP_K.value,
                top_p=ENV.SHELL_TOP_P.value,
                temperature=ENV.SHELL_TEMPERATURE.value,
                stream=True,
            )
            # 本轮助手的完整回复（逐段累积）
            cur_msg = ""
            async for chunk in (await shell_completion(req)).body_iterator:
                # 直接消费响应体的 SSE 流（不经 HTTP 层）
                msg = chunk.decode()  # type: ignore
                # 校验 SSE 帧格式：以 "data: " 开头
                assert msg.startswith("data: "), msg
                # 剥掉 "data: " 前缀（6 个字符）
                msg = msg[6:]
                # 校验帧以换行结尾
                assert msg.endswith("\n"), msg
                # 去掉结尾换行
                msg = msg[:-1]
                if msg == "[DONE]":
                    # 结束标记：不再追加文本
                    continue
                # 累积完整回复
                cur_msg += msg
                # 流式打印（不换行、立即刷新）
                print(msg, end="", flush=True)
            # 本轮回复结束：补一个换行
            print("", flush=True)
            # 记入历史（供下轮构造多轮上下文）
            history.append((cmd, cur_msg))
    except EOFError:
        # user pressed Ctrl-D
        # 用户按 Ctrl-D 退出：静默处理
        pass
    finally:
        # 无论正常退出还是 EOF 都要清理
        print("Exiting shell...")
        # 给在途输出一点收尾时间
        await asyncio.sleep(0.1)
        # 关停前端管理器（ZMQ 队列）
        get_global_state().shutdown()
        # then kill all the subprocesses
        # 强杀所有子进程（调度器/tokenizer），保证不留孤儿进程
        import psutil

        # 取当前进程
        parent = psutil.Process()
        # 递归遍历所有子进程并逐个杀掉
        for child in parent.children(recursive=True):
            child.kill()


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
    """

    # 引用模块级单例
    global _GLOBAL_STATE

    if run_shell:
        # shell 模式必须加载真实权重（要真实对话效果，dummy 权重输出无意义）
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."

    # 服务器监听地址
    host = config.server_host
    # 服务器监听端口
    port = config.server_port

    # 单例只允许初始化一次
    assert _GLOBAL_STATE is None, "Global state is already initialized"
    _GLOBAL_STATE = FrontendManager(
        config=config,
        # PULL 队列：接收 tokenizer 的回包（本端创建 IPC 端点）
        recv_tokenizer=ZmqAsyncPullQueue(
            config.zmq_frontend_addr,
            create=True,
            decoder=BaseFrontendMsg.decoder,
        ),
        # PUSH 队列：把请求发给 tokenizer（连接端由对端创建与否视配置）
        send_tokenizer=ZmqAsyncPushQueue(
            config.zmq_tokenizer_addr,
            create=config.frontend_create_tokenizer_link,
            encoder=BaseTokenizerMsg.encoder,
        ),
    )

    # start the backend here
    # 启动后端子进程（TP 调度器 + tokenizer/detokenizer），由调用方注入回调
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")
    if not run_shell:
        # 常规模式：启动 uvicorn 事件循环对外提供 HTTP 服务（阻塞直到关停）
        uvicorn.run(app, host=host, port=port)
    else:
        # shell 模式：不启 HTTP，直接跑交互式终端
        asyncio.run(shell())
