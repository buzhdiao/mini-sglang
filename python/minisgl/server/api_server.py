from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解使用前向引用

import asyncio  # 异步事件：用于等待/唤醒 token 回复的 event 机制
import json  # 序列化 OpenAI 流式 chunk 为 JSON
import time  # 取时间戳：模型卡 created 字段与响应 created 字段
from contextlib import asynccontextmanager  # 异步上下文管理器装饰器：实现 FastAPI lifespan
from dataclasses import dataclass, field  # dataclass 与 field：声明 FrontendManager 及带默认工厂的字段
from typing import Callable, Dict, List, Literal, Tuple  # 类型注解

import uvicorn  # ASGI 服务器：实际监听 HTTP 端口运行 FastAPI 应用
from fastapi import FastAPI, Request  # Web 框架核心与请求对象（检测客户端断连）
from fastapi.responses import StreamingResponse  # 流式响应：以 SSE 方式逐块推送生成结果
from minisgl.core import SamplingParams  # 采样参数结构：透传给调度器（温度、top_k 等）
from minisgl.env import ENV  # 环境变量配置：shell 模式下的默认采样参数
from minisgl.message import (  # 前端/分词器间通信的消息类型定义
    AbortMsg,  # 发往 tokenizer 的中止请求消息
    BaseFrontendMsg,  # 前端收到的消息基类（提供 decoder）
    BaseTokenizerMsg,  # 前端发出的消息基类（提供 encoder）
    BatchFrontendMsg,  # 前端收到的批量消息包装
    TokenizeMsg,  # 发往 tokenizer 的编码请求消息
    UserReply,  # tokenizer 返回的用户回复（增量文本 + 结束标志）
)
from minisgl.utils import ZmqAsyncPullQueue, ZmqAsyncPushQueue, init_logger  # asyncio 版 ZMQ 收发队列与日志初始化
from prompt_toolkit import PromptSession  # 交互式终端会话（shell 模式用的异步输入框）
from prompt_toolkit.completion import WordCompleter  # 命令自动补全（/exit、/reset）
from pydantic import BaseModel, Field  # Pydantic 模型：请求体校验与 OpenAPI 文档生成
from starlette.background import BackgroundTask  # 响应结束后的后台任务（用于清理中止请求）

from .args import ServerArgs  # 服务器配置（host/port、ZMQ 地址等）

logger = init_logger(__name__, "FrontendAPI")  # 初始化前端 API 日志器

_GLOBAL_STATE = None  # 全局唯一的 FrontendManager 实例（进程级单例，路由函数通过它访问 ZMQ 链路）


def get_global_state() -> FrontendManager:
    """获取全局前端状态；若尚未初始化则断言失败（未调用 run_api_server 就发起请求属于编程错误）"""
    global _GLOBAL_STATE  # 声明使用模块级全局变量
    assert _GLOBAL_STATE is not None, "Global state is not initialized"  # 防御性检查：必须先初始化
    return _GLOBAL_STATE  # 返回全局单例


def _unwrap_msg(msg: BaseFrontendMsg) -> List[UserReply]:
    """把前端收到的消息统一解包成 UserReply 列表（批量消息逐条展开，单条消息包装成列表）"""
    if isinstance(msg, BatchFrontendMsg):  # 批量消息：逐条校验并收集
        result = []
        for reply in msg.data:  # 遍历批内每条回复
            assert isinstance(reply, UserReply)  # 确保类型正确（消息协议契约检查）
            result.append(reply)
        return result
    assert isinstance(msg, UserReply)  # 单条消息：校验类型
    return [msg]  # 包装成单元素列表，统一调用方的迭代逻辑


class GenerateRequest(BaseModel):
    """/generate 接口的请求体：极简的原始文本生成请求"""

    prompt: str  # 用户输入的原始 prompt 文本
    max_tokens: int  # 最大生成 token 数
    ignore_eos: bool = False  # 是否忽略 EOS 继续生成（测试吞吐时常用）


class Message(BaseModel):
    """OpenAI chat 格式的单条对话消息"""

    role: Literal["system", "user", "assistant"]  # 角色只能是三种之一
    content: str  # 消息文本内容


class OpenAICompletionRequest(BaseModel):
    """Unified request model for OpenAI-style completions and chat-completions."""  # 兼容 completions 与 chat/completions 两种请求的统一模型

    model: str  # 模型名（OpenAI 协议必填，本实现未实际校验）

    prompt: str | None = None  # completions 风格的原始文本（与 messages 二选一）
    messages: List[Message] | None = None  # chat 风格的对话消息列表（与 prompt 二选一）

    max_tokens: int = 16  # 最大生成 token 数（OpenAI 默认 16）
    temperature: float = 1.0  # 温度采样参数

    top_k: int = -1  # top-k 采样（-1 表示禁用）
    top_p: float = 1.0  # nucleus 采样阈值（1.0 表示禁用）
    n: int = 1  # 生成候选数（本实现未使用，仅占位）
    stream: bool = False  # 是否流式返回
    stop: List[str] = []  # 停止字符串列表（本实现未使用）
    presence_penalty: float = 0.0  # 存在惩罚（本实现未使用）
    frequency_penalty: float = 0.0  # 频率惩罚（本实现未使用）

    ignore_eos: bool = False  # 扩展字段：忽略 EOS 继续生成


class ModelCard(BaseModel):
    """OpenAI /v1/models 返回的单个模型卡片"""

    id: str  # 模型 id（这里用模型路径）
    object: str = "model"  # OpenAI 协议固定值 "model"
    created: int = Field(default_factory=lambda: int(time.time()))  # 创建时间戳，默认为当前时间
    owned_by: str = "mini-sglang"  # 模型归属方
    root: str  # 根模型名（OpenAI 协议字段）


class ModelList(BaseModel):
    """OpenAI /v1/models 返回的模型列表"""

    object: str = "list"  # OpenAI 协议固定值 "list"
    data: List[ModelCard] = Field(default_factory=list)  # 模型卡片列表


@dataclass  # 数据类装饰器：自动生成 __init__
class FrontendManager:
    """前端管理器：维护 API server 与 tokenizer 进程之间的 ZMQ 链路、请求生命周期与流式回复"""

    config: ServerArgs  # 服务器配置引用（模型路径、地址等）
    send_tokenizer: ZmqAsyncPushQueue[BaseTokenizerMsg]  # 异步 PUSH 队列：把消息发往 tokenizer 进程
    recv_tokenizer: ZmqAsyncPullQueue[BaseFrontendMsg]  # 异步 PULL 队列：接收 tokenizer 回传的回复
    uid_counter: int = 0  # 递增的请求 id 计数器（uid 即请求唯一标识）
    initialized: bool = False  # 后台监听协程是否已创建（只需创建一次）
    ack_map: Dict[int, List[UserReply]] = field(default_factory=dict)  # uid -> 未消费的回复列表（生产者/消费者缓冲）
    event_map: Dict[int, asyncio.Event] = field(default_factory=dict)  # uid -> 唤醒事件（有新回复时 set）

    def new_user(self) -> int:
        """为新请求分配 uid，并创建对应的回复缓冲与唤醒事件"""
        uid = self.uid_counter  # 取当前计数作为新请求的 uid
        self.uid_counter += 1  # 计数器递增，保证 uid 单调唯一
        self.ack_map[uid] = []  # 初始化该请求的回复缓冲
        self.event_map[uid] = asyncio.Event()  # 初始化该请求的唤醒事件
        return uid  # 返回分配的 uid

    async def listen(self):
        """后台监听协程：持续从 ZMQ 收回复并按 uid 分发到缓冲区，唤醒等待者"""
        while True:  # 常驻循环
            msg = await self.recv_tokenizer.get()  # 异步等待下一条回复消息
            for msg in _unwrap_msg(msg):  # 解包批量消息，逐条处理（复用循环变量名 msg）
                if msg.uid not in self.ack_map:  # 该请求已不在等待列表（如已断连被清理），直接丢弃
                    continue
                self.ack_map[msg.uid].append(msg)  # 回复放入缓冲
                self.event_map[msg.uid].set()  # set 事件唤醒该请求的等待协程

    def _create_listener_once(self):
        """惰性创建后台监听协程：第一次发送消息前确保 listen() 已在事件循环中运行"""
        if not self.initialized:  # 尚未创建过
            asyncio.create_task(self.listen())  # 把监听协程挂到当前事件循环
            self.initialized = True  # 标记已创建，避免重复

    async def send_one(self, msg: BaseTokenizerMsg):
        """发送单条消息给 tokenizer（首次发送前会自动启动监听协程）"""
        self._create_listener_once()  # 确保监听协程已就绪（能收到回复）
        await self.send_tokenizer.put(msg)  # 异步推送到 ZMQ 队列

    async def wait_for_ack(self, uid: int):
        """异步生成器：不断产出该 uid 的回复，直到收到 finished 标志；结束时清理该请求的状态"""
        event = self.event_map[uid]  # 取出该请求的唤醒事件

        while True:  # 持续消费直到流结束
            await event.wait()  # 挂起等待新回复到达
            event.clear()  # 立即清零事件，避免重复唤醒

            pending = self.ack_map[uid]  # 取走全部待消费回复
            self.ack_map[uid] = []  # 缓冲清空，供下一轮继续积累
            ack = None
            for ack in pending:  # 逐条产出本批回复
                yield ack
            if ack and ack.finished:  # 本批最后一条带结束标志：流结束
                break

        del self.ack_map[uid]  # 清理回复缓冲
        del self.event_map[uid]  # 清理唤醒事件

    async def stream_generate(self, uid: int):
        """/generate 接口的 SSE 流：每条回复一行 `data: <文本>`，最后发 [DONE]"""
        async for ack in self.wait_for_ack(uid):  # 逐条等待回复
            yield f"data: {ack.incremental_output}\n".encode()  # 编码成 SSE data 帧字节流
            if ack.finished:  # 收到结束标志即停止
                break
        yield "data: [DONE]\n".encode()  # SSE 约定的流结束标记
        logger.debug("Finished streaming response for user %s", uid)  # 调试日志：流结束

    async def stream_chat_completions(self, uid: int):
        """chat/completions 流式接口的 SSE 流：产出符合 OpenAI chunk 格式的增量 JSON"""
        first_chunk = True  # 首个 chunk 需要带上 role 字段
        async for ack in self.wait_for_ack(uid):  # 逐条等待回复
            delta = {}  # 本 chunk 的增量内容
            if first_chunk:  # 第一个 chunk：声明 assistant 角色
                delta["role"] = "assistant"
                first_chunk = False
            if ack.incremental_output:  # 有新文本才放入 content（避免空 delta）
                delta["content"] = ack.incremental_output

            chunk = {  # 构造 OpenAI 流式 chunk 结构
                "id": f"cmpl-{uid}",  # 用 uid 生成响应 id
                "object": "text_completion.chunk",  # chunk 类型标记
                "choices": [{"delta": delta, "index": 0, "finish_reason": None}],  # 单候选，未结束故 finish_reason 为 None
            }
            yield f"data: {json.dumps(chunk)}\n\n".encode()  # 序列化成 SSE 帧（注意这里用双换行分隔）

            if ack.finished:  # 收到结束标志即停止
                break

        # send final finish_reason
        # 发送最后一个 chunk：delta 为空，finish_reason 标记为 stop
        end_chunk = {
            "id": f"cmpl-{uid}",  # 与前序 chunk 保持同一 id
            "object": "text_completion.chunk",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],  # 结束原因固定为 stop
        }
        yield f"data: {json.dumps(end_chunk)}\n\n".encode()  # 发送结束 chunk
        yield b"data: [DONE]\n\n"  # SSE 流结束标记
        logger.debug("Finished streaming response for user %s", uid)  # 调试日志

    async def stream_with_cancellation(self, generator, request: Request, uid: int):
        """包装流式生成器：周期性检测客户端断连，断连时中止后端请求并透传取消异常"""
        try:
            async for chunk in generator:  # 逐块转发底层生成器的内容
                # detect if the client has disconnected
                # 每发一块前检查客户端是否已断开（如用户关闭了页面）
                if await request.is_disconnected():
                    logger.info("Client disconnected for user %s", uid)  # 记录断连
                    raise asyncio.CancelledError  # 主动触发取消，进入下面的清理逻辑
                yield chunk  # 正常转发数据块
        except asyncio.CancelledError:  # 捕获取消（含上面主动抛出的）
            asyncio.create_task(self.abort_user(uid))  # 后台异步通知 tokenizer/调度器中止该请求
            raise  # 重新抛出，让 FastAPI 正常结束响应

    async def abort_user(self, uid: int):
        """中止指定请求：清理本地状态并通知后端停止生成"""
        await asyncio.sleep(0.1)  # 延迟 0.1s：给在途的回复一点时间，避免竞态误删
        if uid in self.ack_map:  # 清理回复缓冲（若还存在）
            del self.ack_map[uid]
        if uid in self.event_map:  # 清理唤醒事件（若还存在）
            del self.event_map[uid]
        logger.warning("Aborting request for user %s", uid)  # 警告日志：请求被中止
        await self.send_one(AbortMsg(uid=uid))  # 把中止消息发给 tokenizer → 调度器

    def shutdown(self):
        """停止两端的 ZMQ 队列（进程退出前的清理）"""
        self.send_tokenizer.stop()  # 停止发送队列
        self.recv_tokenizer.stop()  # 停止接收队列


@asynccontextmanager  # 装饰器：把异步生成器函数变成 async 上下文管理器，供 FastAPI lifespan 使用
async def lifespan(_: FastAPI):
    """FastAPI 生命周期钩子：yield 前是启动逻辑（此处为空），yield 后是关闭逻辑"""
    yield  # 服务运行中（此处不做事）
    # shutdown code here
    global _GLOBAL_STATE  # 声明使用全局变量
    if _GLOBAL_STATE is not None:  # 若全局状态存在
        _GLOBAL_STATE.shutdown()  # 关闭 ZMQ 队列


app = FastAPI(title="MiniSGL API Server", version="0.0.1", lifespan=lifespan)  # FastAPI 应用实例，绑定生命周期钩子


@app.post("/generate")  # 路由：POST /generate（原生极简生成接口）
async def generate(req: GenerateRequest, request: Request):
    """处理 /generate：编码请求发给 tokenizer，然后以 SSE 流式返回生成文本"""
    logger.debug("Received generate request %s", req)  # 调试日志：请求内容
    state = get_global_state()  # 取全局前端状态
    uid = state.new_user()  # 分配新请求 uid
    await state.send_one(  # 把编码请求发给 tokenizer 进程
        TokenizeMsg(
            uid=uid,  # 请求唯一 id
            text=req.prompt,  # 原始文本（走纯文本路径，不套对话模板）
            sampling_params=SamplingParams(  # 构造采样参数
                ignore_eos=req.ignore_eos,  # 是否忽略 EOS
                max_tokens=req.max_tokens,  # 最大生成数
            ),
        )
    )

    return StreamingResponse(  # 返回 SSE 流式响应
        state.stream_with_cancellation(state.stream_generate(uid), request, uid),  # 包装了断连检测的生成流
        media_type="text/event-stream",  # SSE 媒体类型
    )


@app.api_route("/v1", methods=["GET", "POST", "HEAD", "OPTIONS"])  # 路由：/v1 根路径接受多种方法
async def v1_root():
    """健康检查端点：任何对 /v1 的请求都返回 ok"""
    return {"status": "ok"}


@app.post("/v1/chat/completions")  # 路由：OpenAI 兼容的 chat/completions
async def v1_completions(req: OpenAICompletionRequest, request: Request):
    """处理 chat/completions：支持 messages（套对话模板）或 prompt（纯文本），流式/非流式均可"""
    state = get_global_state()  # 取全局前端状态
    if req.messages:  # 有对话消息列表：转成 dict 列表交给 tokenizer 套模板
        prompt = [msg.model_dump() for msg in req.messages]
    else:  # 否则必须是纯文本 prompt
        assert req.prompt is not None, "Either 'messages' or 'prompt' must be provided"  # 两者都缺则报错
        prompt = req.prompt

    # TODO: support more sampling parameters
    # TODO：目前只透传了部分采样参数（stop/penalty 等未支持）
    uid = state.new_user()  # 分配新请求 uid
    await state.send_one(  # 发送编码请求
        TokenizeMsg(
            uid=uid,
            text=prompt,  # 消息列表或纯文本
            sampling_params=SamplingParams(  # 透传采样参数
                ignore_eos=req.ignore_eos,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
            ),
        )
    )

    if req.stream:  # 流式模式：返回 SSE 响应
        return StreamingResponse(
            state.stream_with_cancellation(state.stream_chat_completions(uid), request, uid),  # OpenAI chunk 格式流
            media_type="text/event-stream",  # SSE 媒体类型
        )

    # Non-streaming: collect all chunks and return a single JSON response
    # 非流式模式：等全部生成完毕，聚合成单个 JSON 响应
    full_content = ""  # 完整生成文本
    async for ack in state.wait_for_ack(uid):  # 逐条消费回复
        full_content += ack.incremental_output  # 拼接增量文本
        if ack.finished:  # 结束标志到达
            break

    return {  # 返回 OpenAI chat.completion 格式响应
        "id": f"chatcmpl-{uid}",  # 响应 id
        "object": "chat.completion",  # 对象类型
        "created": int(time.time()),  # 创建时间戳
        "model": req.model,  # 回显请求的模型名
        "choices": [
            {
                "index": 0,  # 单候选
                "message": {"role": "assistant", "content": full_content},  # 完整回复内容
                "finish_reason": "stop",  # 结束原因
            }
        ],
        "usage": {  # token 用量统计（本实现未统计，全部填 0）
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


@app.get("/v1/models")  # 路由：OpenAI 兼容的模型列表
async def available_models():
    """返回当前服务加载的模型卡片列表（单模型）"""
    state = get_global_state()  # 取全局前端状态
    return ModelList(data=[ModelCard(id=state.config.model_path, root=state.config.model_path)])  # 用模型路径作为 id/root


async def shell_completion(req: OpenAICompletionRequest):
    """shell 模式专用的补全请求：与 chat/completions 类似，但内部直接消费 SSE 流且始终流式"""
    state = get_global_state()  # 取全局前端状态
    assert req.messages is not None, "Shell completion only supports chat-completions"  # shell 只支持对话形式
    prompt = [msg.model_dump() for msg in req.messages]  # 消息转 dict 列表（含完整历史）

    # TODO: support more sampling parameters
    # TODO：同上，采样参数仅透传部分
    uid = state.new_user()  # 分配新 uid
    await state.send_one(  # 发送编码请求
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

    async def _abort():  # 定义中止协程（响应结束后由 BackgroundTask 调用）
        await state.abort_user(uid)  # 中止该请求

    return StreamingResponse(  # 返回 SSE 响应
        state.stream_generate(uid),  # 使用 /generate 风格的简单流（shell 自己解析）
        media_type="text/event-stream",
        background=BackgroundTask(lambda: _abort),  # 响应结束后注册清理任务
    )



async def shell():
    """交互式终端 shell：多轮对话、/exit 退出、/reset 清空历史；Ctrl-D 优雅退出"""
    commands = ["/exit", "/reset"]  # 支持的斜杠命令
    completer = WordCompleter(commands)  # 命令补全器
    session = PromptSession("$ ", completer=completer)  # 创建异步输入会话（提示符 $）

    try:
        history: List[Tuple[str, str]] = []  # 对话历史：[(用户输入, 助手回复), ...]
        while True:  # 主循环
            cmd = (await session.prompt_async()).strip()  # 异步读取一行输入并去首尾空白
            if cmd == "":  # 空输入直接跳过
                continue
            if cmd.startswith("/"):  # 斜杠开头视为命令
                if cmd == "/exit":  # 退出命令
                    return
                if cmd == "/reset":  # 清空历史命令
                    history = []
                    continue
                raise ValueError(f"Unknown command: {cmd}")  # 未知命令报错
            history_messages: List[Message] = []  # 把历史对话展开成消息列表
            for user_msg, assistant_msg in history:  # 每轮历史展开为一对 user/assistant 消息
                history_messages.append(Message(role="user", content=user_msg))
                history_messages.append(Message(role="assistant", content=assistant_msg))
            # send to server
            # 构造 chat 请求（历史 + 当前输入），采样参数来自环境变量
            req = OpenAICompletionRequest(
                model="",  # shell 模式不指定模型名
                messages=history_messages + [Message(role="user", content=cmd)],  # 历史在前，当前输入在后
                max_tokens=ENV.SHELL_MAX_TOKENS.value,  # 环境变量配置的最大生成数
                top_k=ENV.SHELL_TOP_K.value,  # 环境变量配置的 top_k
                top_p=ENV.SHELL_TOP_P.value,  # 环境变量配置的 top_p
                temperature=ENV.SHELL_TEMPERATURE.value,  # 环境变量配置的温度
                stream=True,  # 强制流式（边生成边打印）
            )
            cur_msg = ""  # 累积本轮助手回复
            async for chunk in (await shell_completion(req)).body_iterator:  # 直接消费 SSE 响应的原始字节流
                msg = chunk.decode()  # type: ignore  # 字节解码为字符串
                assert msg.startswith("data: "), msg  # 帧格式校验：必须以 "data: " 开头
                msg = msg[6:]  # 去掉 "data: " 前缀
                assert msg.endswith("\n"), msg  # 帧格式校验：以换行结尾
                msg = msg[:-1]  # 去掉末尾换行
                if msg == "[DONE]":  # 流结束标记：跳过（外层循环自然会退出）
                    continue
                cur_msg += msg  # 累积回复文本
                print(msg, end="", flush=True)  # 实时打印增量（不换行、立即刷新）
            print("", flush=True)  # 本轮结束打印换行
            history.append((cmd, cur_msg))  # 本轮对话写入历史
    except EOFError:  # 捕获输入结束
        # user pressed Ctrl-D
        pass  # 用户按 Ctrl-D：静默退出
    finally:  # 无论正常退出还是 Ctrl-D 都执行清理
        print("Exiting shell...")
        await asyncio.sleep(0.1)  # 稍等在途请求完成
        get_global_state().shutdown()  # 关闭 ZMQ 队列
        # then kill all the subprocesses
        # 强制杀掉所有子进程（调度器/tokenizer 等），保证干净退出
        import psutil  # 延迟导入：仅在 shell 退出路径需要

        parent = psutil.Process()  # 当前主进程
        for child in parent.children(recursive=True):  # 递归枚举所有子进程
            child.kill()  # 逐个强杀


def run_api_server(config: ServerArgs, start_backend: Callable[[], None], run_shell: bool) -> None:
    """
    Run the frontend API server (FastAPI + uvicorn) and wire it to the tokenizer process via ZMQ.
    启动前端 API server（FastAPI + uvicorn），并通过 ZMQ 与 tokenizer 进程建立链路。

    Args:
        config: Server configuration (host/port, ZMQ IPC addresses, etc).
            服务器配置（host/port、ZMQ IPC 地址等）。
        start_backend: Callback that launches the backend worker processes (TP schedulers +
            tokenizer/detokenizer).
            启动后端工作子进程（TP 调度器 + tokenizer/detokenizer）的回调。
        run_shell: If True, run an interactive terminal shell instead of starting uvicorn.
            为 True 时运行交互式终端 shell，替代 uvicorn。
    """

    global _GLOBAL_STATE  # 声明使用模块级全局变量

    if run_shell:  # shell 模式校验
        assert not config.use_dummy_weight, "Shell mode does not support dummy weights."  # shell 模式必须加载真实权重（要生成真实文本）

    host = config.server_host  # 监听地址
    port = config.server_port  # 监听端口

    assert _GLOBAL_STATE is None, "Global state is already initialized"  # 防止重复初始化
    _GLOBAL_STATE = FrontendManager(  # 创建全局前端状态
        config=config,
        recv_tokenizer=ZmqAsyncPullQueue(  # 接收队列：前端从 tokenizer 收回复
            config.zmq_frontend_addr,  # 前端侧地址（由前端 bind，create=True）
            create=True,
            decoder=BaseFrontendMsg.decoder,  # 按消息协议解码
        ),
        send_tokenizer=ZmqAsyncPushQueue(  # 发送队列：前端向 tokenizer 发请求
            config.zmq_tokenizer_addr,  # tokenizer 侧地址（共享模式下与 detokenizer 同址，由对方 bind）
            create=config.frontend_create_tokenizer_link,  # 仅独立 tokenizer 模式下由前端 bind
            encoder=BaseTokenizerMsg.encoder,  # 按消息协议编码
        ),
    )

    # start the backend here
    # 在 uvicorn 启动前先拉起后端子进程（调度器 + tokenizer/detokenizer）
    start_backend()

    logger.info(f"API server is ready to serve on {host}:{port}")  # 就绪日志
    if not run_shell:  # 正常模式
        uvicorn.run(app, host=host, port=port)  # 阻塞运行 ASGI 服务器（主进程事件循环由 uvicorn 接管）
    else:  # shell 模式
        asyncio.run(shell())  # 用 asyncio 运行交互式 shell
