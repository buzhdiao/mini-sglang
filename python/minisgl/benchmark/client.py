# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# asyncio 并发发送请求
import asyncio
# random 生成随机 token id
import random
# time.perf_counter 记录高精度时间戳
import time
# contextmanager 定义 with 上下文管理器
from contextlib import contextmanager
# dataclass/field 声明数据类
from dataclasses import dataclass, field
# 类型注解工具
from typing import Any, Dict, List, Tuple, overload

# UNSET/Unset 用于区分"未传参"与 None；init_logger 创建 logger
from minisgl.utils import UNSET, Unset, init_logger
# OpenAI 兼容异步客户端
from openai import AsyncOpenAI as OpenAI
# pydantic BaseModel 校验轨迹 JSON 结构
from pydantic import BaseModel
# tqdm 异步友好的进度条
from tqdm.asyncio import tqdm

# 创建模块级 logger
logger = init_logger(__name__)


# 一条回放轨迹：到达时间 + 消息 + 输入/输出长度（frozen 保证不可变）
@dataclass(frozen=True)
class BenchmarkTrace:
    # 请求到达时间戳（单位秒）
    timestamp: float  # unit (second)
    # 发送的 prompt 文本
    message: str
    # 期望输出 token 数
    output_length: int  # output length in tokens
    # 输入 token 数（可选，用于结果统计）
    input_length: int | None = None  # input length in tokens, optional


# 单条请求的基准结果：时间戳序列 + 输入/输出长度
@dataclass(frozen=True)
class BenchOneResult:
    # 从请求发出到每个响应块到达的 perf_counter 时间戳
    tics: List[float]
    # 输入 token 数
    input_len: int
    # 输出 token 数
    output_len: int

    # 序列化为 JSON：前两个是输入/输出长度，后接全部时间戳
    def as_json(self) -> List[float]:
        # 拼接成一行可持久化的浮点数组
        return [self.input_len, self.output_len] + self.tics

    # 从 JSON 反序列化回来
    @staticmethod
    def from_json(raw: List[float]) -> BenchOneResult:
        # check raw[0] and raw[1] are integers
        # 输入/输出长度必须是整数（时间戳是浮点）
        assert raw[0].is_integer() and raw[1].is_integer()
        # 还原对象：前两位转 int，其余是时间戳
        return BenchOneResult(tics=raw[2:], input_len=int(raw[0]), output_len=int(raw[1]))


# 处理过程中的原始结果（尚未转成可序列化形式）
@dataclass(frozen=True)
class RawResult:
    # 输入 token 数（可能未知，None 时由 tokenizer 补算）
    input_len: int | None
    # 输出 token 数
    output_len: int
    # 发送的 prompt
    message: str
    # 时间戳序列
    tics: List[float]


# 简单计数器：记录当前值与历史峰值（并发/排队统计用）
@dataclass
class Counter:
    # 当前计数
    current: int = 0
    # 历史最大值
    history_max: int = 0

    # 增加 n 并刷新历史峰值
    def inc(self, n=1):
        # 当前值累加
        self.current += n
        # 更新历史最大
        self.history_max = max(self.history_max, self.current)

    # 减少 n，且不允许减到负数
    def dec(self, n=1):
        # 当前值递减
        self.current -= n
        # 防御性断言：计数不应为负
        assert self.current >= 0


# 压测过程的控制台：4 个进度条 + inflight/queue 计数器
@dataclass
class Console:
    # 已发送请求数进度条
    input_pbar: tqdm
    # 已完成请求数进度条
    output_pbar: tqdm
    # prefill token 进度条
    prefill_pbar: tqdm
    # decode token 进度条
    decode_pbar: tqdm
    # 是否禁用进度条输出
    disabled: bool
    # 在途请求计数器
    inflight_counter: Counter = field(default_factory=Counter)
    # 排队请求计数器（已发出、还没开始 prefill）
    queue_counter: Counter = field(default_factory=Counter)

    # 请求发出时更新：input 进度条 + inflight/queue 计数
    def update_input(self, n=1):
        # 推进"已发送"进度条
        self.input_pbar.update(n)
        # 立即刷新显示
        self.input_pbar.refresh()
        # 在途请求数增加
        self.inflight_counter.inc(n)
        # 排队请求数增加
        self.queue_counter.inc(n)

    # 请求完成时更新：output 进度条 + inflight 计数回退
    def update_output(self, n=1):
        # 推进"已完成"进度条
        self.output_pbar.update(n)
        # 立即刷新显示
        self.output_pbar.refresh()
        # 在途请求数减少
        self.inflight_counter.dec(n)

    # 收到首个响应块（prefill 结束）时更新：prefill token 进度 + queue 计数回退
    def update_prefill(self, n=1):
        # 推进 prefill token 进度条
        self.prefill_pbar.update(n)
        # 立即刷新显示
        self.prefill_pbar.refresh()
        # 该请求已离开排队状态
        self.queue_counter.dec(n)

    # 每收到一个 decode 块更新 decode token 进度
    def update_decode(self, n=1):
        # 推进 decode token 进度条
        self.decode_pbar.update(n)

    # 上下文管理器：进入时记录请求发出，退出时记录请求完成
    @contextmanager
    def inflight(self, n=1):
        # 进入：算作已发送
        self.update_input(n)
        # 让出控制权给被包裹的代码块
        yield
        # 退出：算作已完成
        self.update_output(n)

    # 上下文管理器：退出时关闭所有进度条并打印并发/排队峰值
    @contextmanager
    def log_stats(self):
        # 让出控制权给压测主体
        yield
        # 关闭各进度条，恢复终端显示
        self.input_pbar.close()
        # 关闭"已完成"进度条
        self.output_pbar.close()
        # 关闭 prefill 进度条
        self.prefill_pbar.close()
        # 关闭 decode 进度条
        self.decode_pbar.close()
        # 仅在启用进度条时打印统计
        if not self.disabled:
            # 取历史峰值：最大在途请求数
            max_inflight = self.inflight_counter.history_max
            # 取历史峰值：最大排队请求数
            max_queue = self.queue_counter.history_max
            # 打印并发与排队峰值
            logger.info(f"Max inflight requests: {max_inflight}, Max queued requests: {max_queue}")


# 一批压测结果的容器，支持 JSON 序列化/反序列化
@dataclass(frozen=True)
class BenchmarkResult:
    # 每条请求的结果列表
    raw_data: List[BenchOneResult]

    # 整体序列化成二维数组
    def as_json(self) -> List[List[float]]:
        # 逐条转换
        return [r.as_json() for r in self.raw_data]

    # 从二维数组还原
    @staticmethod
    def from_json(raw: List[List[float]]) -> BenchmarkResult:
        # 逐条反序列化
        return BenchmarkResult(raw_data=[BenchOneResult.from_json(r) for r in raw])


# 创建控制台：按请求/token 总量初始化 4 个进度条并调整对齐格式
def make_console(num_requests: int, sum_output_length: int, use_pbar: bool = True) -> Console:
    # 基础进度条格式：描述 + 百分比 + 计数 + 速率/耗时
    BAR_FORMAT_0 = (
        "{desc:<10} {percentage:3.0f}%|{bar}|"
        " {n_fmt:>5}/{total_fmt} "
        "[{rate_fmt:>12} {elapsed:>8}/{remaining:<8}]"
    )
    # decode 条初始与基础格式相同
    BAR_FORMAT_1 = BAR_FORMAT_0
    # 计数列初始宽度
    n_fmt_align = 5
    # prefill token 总数：每请求按 1 个 token 计（首 token 即 prefill 完成）
    prefill_tokens = num_requests
    # decode token 总数 = 输出预算 - prefill 部分
    decode_tokens = sum_output_length - prefill_tokens

    # decode 总数位数更宽时扩展格式宽度
    if len(str(decode_tokens)) > n_fmt_align:
        # 按最长数字宽度对齐
        n_fmt_align = len(str(decode_tokens))
        # 同步替换格式串中的宽度
        BAR_FORMAT_0 = BAR_FORMAT_0.replace("{n_fmt:>5}", "{n_fmt:>" + str(n_fmt_align) + "}")
        # decode 条跟随新格式
        BAR_FORMAT_1 = BAR_FORMAT_0

    # prefill 与 decode 数字宽度不同时，分别调整各自格式，让两列对齐
    if len(str(prefill_tokens)) < len(str(decode_tokens)):
        # 记录旧宽度占位符
        old_align_str = "{n_fmt:>" + str(n_fmt_align) + "}"
        # 加宽对齐量
        n_fmt_align += len(str(decode_tokens)) - len(str(prefill_tokens))
        # 仅调整基础格式（prefill 条用）
        BAR_FORMAT_0 = BAR_FORMAT_0.replace(old_align_str, "{n_fmt:>" + str(n_fmt_align) + "}")

    # 是否禁用进度条
    disabled = not use_pbar
    # "已发送请求"进度条
    input_pbar = tqdm(
        total=num_requests,
        desc="Requests sent",
        position=0,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    # "已完成请求"进度条
    output_pbar = tqdm(
        total=num_requests,
        desc="Requests done",
        position=1,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    # "prefill token"进度条（总数 = 请求数）
    prefill_pbar = tqdm(
        total=prefill_tokens,
        desc="Prefill token",
        position=2,
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    # "decode token"进度条（总数 = 输出预算 - 请求数）
    decode_pbar = tqdm(
        total=decode_tokens,
        desc="Decode token ",
        position=3,
        bar_format=BAR_FORMAT_1,
        disable=disabled,
    )
    # 组装控制台对象
    return Console(
        input_pbar=input_pbar,
        output_pbar=output_pbar,
        prefill_pbar=prefill_pbar,
        decode_pbar=decode_pbar,
        disabled=disabled,
    )


# 生成约为 n 个 token 的随机 prompt（通过编码/解码往返逼近目标长度）
def generate_prompt(tokenizer: Any, n: int) -> str:
    """Generate a prompt of approximately `n` tokens using the provided tokenizer."""
    # 取半个词表范围，避开特殊/罕见 token
    vocab_size = tokenizer.vocab_size // 2
    # 随机生成 n 个 token id
    token_ids = [random.randint(0, vocab_size) for _ in range(n)]

    # 最多迭代 64 轮修整长度（解码再编码可能改变长度）
    for _ in range(64):
        # 先解码成文本
        prompt = tokenizer.decode(token_ids)
        # 再重新编码（add_special_tokens=False 避免附加 BOS 等）
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # 长度恰好达标则返回
        if len(token_ids) == n:
            # 返回最终文本
            return prompt
        # 偏短则补随机 token
        if len(token_ids) < n:
            # 计算还差多少
            need = n - len(token_ids)
            # 补足差额
            token_ids.extend([random.randint(0, vocab_size) for _ in range(need)])
        else:
            # 偏长则截断到 n
            token_ids = token_ids[:n]

    # 64 轮仍无法收敛则报错
    raise ValueError("Failed to generate a message of the desired length.")


# 发送单条流式请求并记录 TTFT/每个 token 的到达时间戳。
# pbar 可传 bool（自动创建单请求 Console）或现成 Console；
# extra_body 额外透传服务端参数；input_length 强制指定输入长度（压测对齐用）
async def benchmark_one(
    client: OpenAI,
    prompt: str,
    output_length: int,
    model: str,
    *,
    pbar: Console | bool = True,
    extra_body: Dict[str, Any] | None = None,
    input_length: int | None = None,  # a hack to force input length
) -> RawResult:
    # 传入 bool 时为单条请求临时建一套进度条
    if isinstance(pbar, bool):
        # 总输出预算按该请求输出长度计
        pbar = make_console(1, output_length, use_pbar=pbar)
    # 进入"在途"统计：发送/完成各计一次
    with pbar.inflight(1):
        # 组装内部扩展参数：ignore_eos 让输出长度可精确控制；top_k=1 即贪心解码
        kwargs = {
            "ignore_eos": True,
            "top_k": 1,
        }
        # this is an internal kwargs that might work for our system
        # 需要时透传输入长度覆盖参数（minisgl 服务端支持）
        if input_length is not None:
            # 写入覆盖项
            kwargs["input_length_override"] = input_length
        # 允许调用方覆盖上面的默认 kwargs
        kwargs.update(extra_body or {})  # can override kwargs
        # 发起流式 chat 请求
        response = await client.chat.completions.create(
            model=model,
            stream=True,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_tokens=output_length,
            temperature=0.0,
            extra_body=kwargs,
        )
        # 先记录请求发出时刻
        tics = [time.perf_counter()]
        # 逐块接收流式响应
        async for _ in response:
            # 每收到一块记一个时间戳
            tics.append(time.perf_counter())
            # 第一个响应块：prefill 完成（TTFT）
            if len(tics) == 2:
                # 更新 prefill 进度
                pbar.update_prefill()
            elif len(tics) <= output_length + 1:
                # 其余块按 decode token 计
                pbar.update_decode()
        # 返回原始结果（tics[0]=发送，tics[1]=TTFT，其余逐 token）
        return RawResult(
            input_len=input_length,
            output_len=output_length,
            message=prompt,
            tics=tics,
        )


# 并发发送一批请求（asyncio.gather），共用同一套进度条
async def benchmark_one_batch(
    client: OpenAI,
    prompts: List[str],
    output_lengths: List[int] | int,
    model: str,
    *,
    extra_body: Dict[str, Any] | None = None,
    input_lengths: List[int | None] | None = None,
    pbar: Console | bool = True,
) -> List[RawResult]:
    # 传入单个 int 时广播为每条请求相同的输出长度
    if isinstance(output_lengths, int):
        # 复制成与 prompts 等长的列表
        output_lengths = [output_lengths] * len(prompts)
    # 传入 bool 时按批量规模创建 Console
    if isinstance(pbar, bool):
        # 输出预算为全部请求之和
        pbar = make_console(len(prompts), sum(output_lengths), use_pbar=pbar)
    # 未指定输入长度时填充 None 列表（行尾注释：绕过类型检查的临时处理）
    if input_lengths is None:
        # 构造 None 列表
        l: List[int | None] = [None] * len(prompts)
        # 赋回变量（绕过 mypy 的类型推断问题）
        input_lengths = l  # work-around for typing bug

    # 为每条请求创建异步任务（zip strict=True 保证三个列表严格等长，不一致时报错）
    tasks = [
        benchmark_one(
            client=client,
            prompt=prompt,
            output_length=output_length,
            model=model,
            pbar=pbar,
            extra_body=extra_body,
            input_length=input_length,
        )
        for prompt, output_length, input_length in zip(
            prompts, output_lengths, input_lengths, strict=True
        )
    ]
    # 包裹 log_stats：结束时打印并发/排队统计
    with pbar.log_stats():
        # 并发等待全部请求完成
        return await asyncio.gather(*tasks)


# 按轨迹时间戳回放一批请求（模拟真实负载的到达过程）
async def benchmark_trace(
    client: OpenAI,
    msgs: List[BenchmarkTrace],
    model: str,
    *,
    pbar: Console | bool = True,
) -> List[RawResult]:
    # 传入 bool 时按轨迹规模创建 Console
    if isinstance(pbar, bool):
        # 输出预算为全部轨迹之和
        sum_output_len = sum(msg.output_length for msg in msgs)
        # 创建控制台
        pbar = make_console(len(msgs), sum_output_len, use_pbar=pbar)
    # 记录回放的起始时刻
    start = time.perf_counter()
    # 用最小时间戳做偏移，使第一条请求立即发出
    offset = min(msg.timestamp for msg in msgs) - 1

    # 定义按时间调度的单请求协程
    async def benchmark_timed(msg: BenchmarkTrace):
        # 计算该请求的目标发出时刻
        target = start + msg.timestamp - offset
        # 距目标还有余量则休眠等待（保持到达节奏）
        await asyncio.sleep(max(0, target - time.perf_counter()))
        # 到点后发送该请求
        return await benchmark_one(
            client, msg.message, msg.output_length, model, pbar=pbar, input_length=msg.input_length
        )

    # 为每条轨迹创建定时任务
    tasks = [benchmark_timed(msg) for msg in msgs]
    # 包裹 log_stats 打印统计
    with pbar.log_stats():
        # 并发执行全部定时任务
        return await asyncio.gather(*tasks)


# 重载签名：带 tokenizer 时返回可序列化结果
@overload
def process_benchmark_results(raw_data: List[RawResult], tokenizer: Any) -> BenchmarkResult: ...


# 重载签名：不带 tokenizer 时仅打印统计、返回 None
@overload
def process_benchmark_results(raw_data: List[RawResult]) -> None: ...


# 汇总打印压测指标：TTFT/TPOT/E2E 的均值与分位数、吞吐等。
# tokenizer 未传时用 UNSET 哨兵与 None 区分
def process_benchmark_results(
    raw_data: List[RawResult],
    tokenizer: Any = UNSET,
) -> BenchmarkResult | None:
    # 收集所有请求的逐 token 间隔（TPOT 样本）
    accum_times: List[float] = []
    # 收集每条请求的首 token 延迟（TTFT 样本）
    first_times: List[float] = []
    # 取出每条请求的时间戳序列
    results = [r.tics for r in raw_data]
    # 逐请求计算间隔
    for tics in results:
        # 本请求相邻时间戳的差值
        deltas: List[float] = []
        # 遍历相邻对
        for i in range(len(tics) - 1):
            # 计算相邻两块的间隔
            diff = tics[i + 1] - tics[i]
            # 记录间隔
            deltas.append(diff)
        # 第一个间隔是发送到首块的延迟 = TTFT
        first_times.append(deltas[0])
        # 其余间隔都是逐 token 生成延迟，并入 TPOT 样本池
        accum_times.extend(deltas[1:])

    # 端到端延迟 = 末块时间 - 发出时间
    e2e_times = [tics[-1] - tics[0] for tics in results]
    # 排序便于取分位数
    first_times.sort()
    # 排序 TPOT 样本
    accum_times.sort()
    # 排序 E2E 样本
    e2e_times.sort()

    # 内部工具：计算均值与 p50/p90/p99/max，scale 用于秒转毫秒
    def _print_stats(times: List[float], scale: float = 1.0) -> Tuple[float, ...]:
        # 至少要有一个样本
        assert len(times) > 0
        # 返回五元组统计量
        return (
            scale * sum(times) / len(times),  # avg
            scale * times[int(len(times) * 0.5)],  # p50
            scale * times[int(len(times) * 0.9)],  # p90
            scale * times[int(len(times) * 0.99)],  # p99
            scale * max(times),  # max
        )

    # 内部工具：按数值大小选择打印精度与宽度
    def _fmt(x: float) -> str:
        # 千位以上取整显示
        if x >= 1000:
            # 6 位宽整数
            return f"{int(x):>6}"
        elif x >= 10:
            # 两位小数
            return f"{x:>6.2f}"
        else:
            # 四位小数
            return f"{x:>6.4f}"

    # 计算 TTFT 统计（秒转毫秒）
    avg_ttft, p50_ttft, p90_ttft, p99_ttft, max_ttft = _print_stats(first_times, 1000)
    # 计算 TPOT 统计（秒转毫秒）
    avg_tpot, p50_tpot, p90_tpot, p99_tpot, max_tpot = _print_stats(accum_times, 1000)
    # 计算 E2E 统计（保持秒）
    avg_e2e, p50_e2e, p90_e2e, p99_e2e, max_e2e = _print_stats(e2e_times)

    # 全体时间戳的最早时刻
    min_time = min(min(r) for r in results)
    # 全体时间戳的最晚时刻
    max_time = max(max(r) for r in results)
    # 压测总时长
    dur = max_time - min_time
    # 时长必须为正（防止除零/异常数据）
    assert dur > 0, "Duration must be positive"

    # 总响应块数 ≈ 总 token 数
    num_tokens = sum(len(tic) for tic in results)
    # 请求数
    num_requests = len(results)

    # 打印请求/ token 总量
    logger.info(f"Num requests: #{num_requests}, Num tokens: #{num_tokens}")
    # 打印 TTFT 统计
    logger.info(
        f"TTFT: {_fmt(avg_ttft)} ms (p50: {_fmt(p50_ttft)} ms, p90: {_fmt(p90_ttft)} ms,"
        f" p99: {_fmt(p99_ttft)} ms, max: {_fmt(max_ttft)} ms)"
    )
    # 打印 TPOT 统计
    logger.info(
        f"TPOT: {_fmt(avg_tpot)} ms (p50: {_fmt(p50_tpot)} ms, p90: {_fmt(p90_tpot)} ms,"
        f" p99: {_fmt(p99_tpot)} ms, max: {_fmt(max_tpot)} ms)"
    )
    # 打印 E2E 统计
    logger.info(
        f"E2E:  {_fmt(avg_e2e) }  s (p50: {_fmt(p50_e2e) }  s, p90: {_fmt(p90_e2e) }  s,"
        f" p99: {_fmt(p99_e2e) }  s, max: {_fmt(max_e2e) }  s)"
    )
    # 打印总时长
    logger.info(f"Duration: {_fmt(dur)} s")
    # 打印 token 吞吐与请求吞吐
    logger.info(f"Throughput: {_fmt(num_tokens / dur)} token/s, {_fmt(num_requests / dur)} req/s")

    # normalize the time to start from zero
    # 把各请求时间戳平移到从 0 开始，便于后续对比/绘图
    results = [[r - min_time for r in tics] for tics in results]
    # 未传 tokenizer：仅打印统计，返回 None
    if isinstance(tokenizer, Unset):
        # 返回空结果
        return None

    # 组装可序列化结果；输入长度缺失时用 tokenizer 现算
    return BenchmarkResult(
        raw_data=[
            BenchOneResult(
                tics=r.tics,
                input_len=(
                    r.input_len
                    if r.input_len is not None
                    else len(tokenizer.encode(r.message, add_special_tokens=False))
                ),
                output_len=r.output_len,
            )
            for r in raw_data
        ]
    )


# 读取 Qwen 线上使用轨迹 jsonl，转成回放轨迹列表。
# n 限制最多读取的行数；dummy 模式用随机文本按长度切片，避免逐条生成
def read_qwen_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    # 定义轨迹行结构并做 pydantic 校验
    class JSONInput(BaseModel):
        # 会话 id
        chat_id: int
        # 父会话 id
        parent_chat_id: int
        # 到达时间戳（秒）
        timestamp: float
        # 输入 token 数
        input_length: int
        # 输出 token 数
        output_length: int
        # type: unused
        # 保留字段，当前未用
        type: str  # unused
        # turn: unused
        # 保留字段，当前未用
        turn: int  # unused
        # hash_ids: unused
        # 前缀哈希（缓存研究用），此处未用
        hash_ids: List[int]  # unused

    # 读取并按需截断行
    with open(file_path, "r") as f:
        # 一次性读入全部行
        lines = f.readlines()
        # 限制行数
        if n is not None:
            # 只取前 n 行
            lines = lines[:n]
    # 逐行校验成结构化对象
    objs = [JSONInput.model_validate_json(line) for line in lines]
    # dummy 模式：先生成一段最长输入，再按每条的长度切片
    if dummy:
        # 生成最长长度的一段随机文本
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        # 编码成 token id
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # 按目标长度切前缀并解码，速度快
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])
    else:
        # 精确模式：为每条轨迹独立生成指定长度的 prompt
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)
    # 转成回放轨迹列表：时间戳/prompt/输入输出长度逐条映射
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


# 读取 Mooncake 格式轨迹（时间戳为毫秒，需换算成秒）
def read_mooncake_trace(
    file_path: str,
    tokenizer: Any,
    n: int | None = None,
    dummy: bool = False,
) -> List[BenchmarkTrace]:
    # Mooncake 轨迹行结构
    class JSONInput(BaseModel):
        # 到达时间戳（毫秒）
        timestamp: int
        # 输入 token 数
        input_length: int
        # 输出 token 数
        output_length: int
        # hash_ids: unused for now
        # 前缀哈希，当前未用
        hash_ids: List[int]  # unused for now

    # 读取并按需截断
    with open(file_path, "r") as f:
        # 读入全部行
        lines = f.readlines()
        # 截断到 n 行
        if n is not None:
            # 只保留前 n 行
            lines = lines[:n]
    # 逐行校验
    objs = [JSONInput.model_validate_json(line) for line in lines]
    # dummy 模式：切片共享一段长文本
    if dummy:
        # 生成最长长度文本
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))
        # 编码为 id
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # 按长度切片解码
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])
    else:
        # 精确生成每条 prompt
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)
    # 转成回放轨迹：时间戳除以 1000 换算成秒，其余字段逐条映射
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp / 1000,
            message=_get_prompt(obj),
            input_length=obj.input_length,
            output_length=obj.output_length,
        )
        for obj in objs
    ]


# 把轨迹的时间轴按 scale 缩放（scale<1 负载更密集），并归一化到从 0 开始
def scale_traces(
    traces: List[BenchmarkTrace],
    scale: float,
) -> List[BenchmarkTrace]:
    # 取最早时间戳作为基准
    min_tic = min(trace.timestamp for trace in traces)
    # 平移到 0 后按 scale 缩放得到新的到达间隔，再按新时间戳升序排序
    return sorted(
        [
            BenchmarkTrace(
                timestamp=(trace.timestamp - min_tic) * scale,
                message=trace.message,
                input_length=trace.input_length,
                output_length=trace.output_length,
            )
            for trace in traces
        ],
        key=lambda x: x.timestamp,
    )


# 通过 /v1/models 查询服务端模型名
async def get_model_name(client: OpenAI) -> str:
    # 取模型列表的第一个
    async for model in client.models.list():
        # 返回模型 id
        return model.id
    # 列表为空时报错
    raise ValueError("No models available")
