from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许在注解中使用前向引用类型

import asyncio  # 导入 asyncio，用于并发压测与按时间戳调度请求
import random  # 导入 random，用于随机采样 token id 构造测试 prompt
import time  # 导入 time，用高精度计时器记录请求各阶段时间戳
from contextlib import contextmanager  # 导入 contextmanager，把生成器函数包装成 with 上下文管理器
from dataclasses import dataclass, field  # 导入 dataclass 装饰器与 field（带默认工厂的字段）
from typing import Any, Dict, List, Tuple, overload  # 导入类型注解工具与函数重载标记

from minisgl.utils import UNSET, Unset, init_logger  # 导入哨兵类型 Unset/单例 UNSET（区分"未传参"与 None）与日志工具
from openai import AsyncOpenAI as OpenAI  # 导入 OpenAI 异步客户端，作为压测的 HTTP 客户端
from pydantic import BaseModel  # 导入 pydantic 基类，用于 trace JSON 行的结构化校验
from tqdm.asyncio import tqdm  # 导入异步友好的 tqdm 进度条

logger = init_logger(__name__)  # 创建本模块 logger


@dataclass(frozen=True)  # 不可变数据类：字段创建后不可修改
class BenchmarkTrace:
    """一条压测请求的描述：到达时间戳、消息内容与期望输入/输出长度。"""

    timestamp: float  # unit (second)  # 请求到达时间（秒），回放时按它控制发送节奏
    message: str  # 请求的 prompt 文本
    output_length: int  # output length in tokens  # 期望输出 token 数
    input_length: int | None = None  # input length in tokens, optional  # 输入 token 数（可选，未提供则事后用分词器统计）


@dataclass(frozen=True)  # 不可变数据类
class BenchOneResult:
    """单个请求的压测结果：时间戳序列与输入/输出长度，支持 JSON 序列化往返。"""

    tics: List[float]  # 时间戳序列：首元素为请求发出时刻，其后每个元素对应收到一个流式 chunk 的时刻
    input_len: int  # 输入 token 数
    output_len: int  # 期望输出 token 数

    def as_json(self) -> List[float]:
        """序列化为 [input_len, output_len, tics...] 的扁平浮点列表。"""
        return [self.input_len, self.output_len] + self.tics  # 长度信息放头部，后接时间戳数组

    @staticmethod  # 静态工厂方法，无需实例即可调用
    def from_json(raw: List[float]) -> BenchOneResult:
        """从 as_json 的扁平列表反序列化回 BenchOneResult。"""
        # check raw[0] and raw[1] are integers  # 校验前两个元素是整数（长度信息）
        assert raw[0].is_integer() and raw[1].is_integer()  # 浮点列表中前两位必须可无损还原为整数
        return BenchOneResult(tics=raw[2:], input_len=int(raw[0]), output_len=int(raw[1]))  # 切出时间戳并还原两个长度字段


@dataclass(frozen=True)  # 不可变数据类
class RawResult:
    """单个请求的原始压测记录（输入长度可能未知，留待事后补齐）。"""

    input_len: int | None  # 输入长度（None 表示未知）
    output_len: int  # 输出长度
    message: str  # prompt 原文
    tics: List[float]  # 请求各阶段时间戳序列


@dataclass  # 可变数据类
class Counter:
    """简单计数器：维护当前值并记录历史峰值。"""

    current: int = 0  # 当前计数值
    history_max: int = 0  # 历史最大值（峰值）

    def inc(self, n=1):
        """计数加 n，并刷新历史峰值。"""
        self.current += n  # 累加当前值
        self.history_max = max(self.history_max, self.current)  # 峰值取较大者

    def dec(self, n=1):
        """计数减 n，断言不允许减成负数。"""
        self.current -= n  # 递减当前值
        assert self.current >= 0  # 非负断言：多减说明计数逻辑有 bug，尽早暴露


@dataclass  # 数据类
class Console:
    """压测过程的终端输出面板：4 条进度条 + 在途/排队计数器。"""

    input_pbar: tqdm  # "已发送请求"进度条
    output_pbar: tqdm  # "已完成请求"进度条
    prefill_pbar: tqdm  # prefill token 进度条
    decode_pbar: tqdm  # decode token 进度条
    disabled: bool  # 是否禁用进度条输出（脚本化运行时关闭以减少干扰）
    inflight_counter: Counter = field(default_factory=Counter)  # 在途请求计数器
    queue_counter: Counter = field(default_factory=Counter)  # 排队计数器（已发送但未完成 prefill）

    def update_input(self, n=1):
        """有 n 个请求发出：推进发送进度条并累加两组计数器。"""
        self.input_pbar.update(n)  # 发送进度条 +n
        self.input_pbar.refresh()  # 立即刷新终端显示
        self.inflight_counter.inc(n)  # 在途计数 +n
        self.queue_counter.inc(n)  # 排队计数 +n（尚未开始 prefill）

    def update_output(self, n=1):
        """有 n 个请求完成：推进完成进度条并减少在途计数。"""
        self.output_pbar.update(n)  # 完成进度条 +n
        self.output_pbar.refresh()  # 立即刷新
        self.inflight_counter.dec(n)  # 在途计数 -n

    def update_prefill(self, n=1):
        """有 n 个请求完成 prefill（收到首个 token）。"""
        self.prefill_pbar.update(n)  # prefill 进度条 +n
        self.prefill_pbar.refresh()  # 立即刷新
        self.queue_counter.dec(n)  # 该请求已离开排队状态

    def update_decode(self, n=1):
        """有 n 个 decode token 生成。"""
        self.decode_pbar.update(n)  # decode 进度条 +n

    @contextmanager  # 装饰器：把该方法变成 with 上下文管理器
    def inflight(self, n=1):
        """请求生命周期上下文：进入时计为发送，退出时计为完成。"""
        self.update_input(n)  # 进入：记录请求发出
        yield  # 挂起点，with 体内执行实际请求
        self.update_output(n)  # 退出：记录请求完成

    @contextmanager  # 装饰器：上下文管理器
    def log_stats(self):
        """压测批次上下文：退出时关闭全部进度条并打印峰值统计。"""
        yield  # with 体内执行整批压测
        self.input_pbar.close()  # 关闭发送进度条
        self.output_pbar.close()  # 关闭完成进度条
        self.prefill_pbar.close()  # 关闭 prefill 进度条
        self.decode_pbar.close()  # 关闭 decode 进度条
        if not self.disabled:  # 未禁用时才打印统计
            max_inflight = self.inflight_counter.history_max  # 在途请求峰值
            max_queue = self.queue_counter.history_max  # 排队请求峰值
            logger.info(f"Max inflight requests: {max_inflight}, Max queued requests: {max_queue}")  # 打印峰值：反映服务端并发与排队程度


@dataclass(frozen=True)  # 不可变数据类
class BenchmarkResult:
    """整批压测的最终结果集合，支持 JSON 序列化/反序列化（便于保存与离线分析）。"""

    raw_data: List[BenchOneResult]  # 每个请求的结果列表

    def as_json(self) -> List[List[float]]:
        """整体序列化为二维浮点列表。"""
        return [r.as_json() for r in self.raw_data]  # 逐条序列化

    @staticmethod  # 静态工厂方法
    def from_json(raw: List[List[float]]) -> BenchmarkResult:
        """从 as_json 输出反序列化回 BenchmarkResult。"""
        return BenchmarkResult(raw_data=[BenchOneResult.from_json(r) for r in raw])  # 逐条还原


def make_console(num_requests: int, sum_output_length: int, use_pbar: bool = True) -> Console:
    """创建压测控制台：按请求数/token 数初始化 4 条计数列对齐的进度条。"""
    BAR_FORMAT_0 = (  # 进度条显示格式：描述 + 百分比 + 进度条 + 计数 + [速率 耗时/剩余]
        "{desc:<10} {percentage:3.0f}%|{bar}|"
        " {n_fmt:>5}/{total_fmt} "
        "[{rate_fmt:>12} {elapsed:>8}/{remaining:<8}]"
    )
    BAR_FORMAT_1 = BAR_FORMAT_0  # 第二种格式初始与 0 相同（供 decode 条使用，可能被对齐调整改写）
    n_fmt_align = 5  # 计数字段的对齐宽度，初始 5 位
    prefill_tokens = num_requests  # prefill 阶段每请求至少产出 1 个首 token，故总量 = 请求数
    decode_tokens = sum_output_length - prefill_tokens  # decode 阶段 token 总量 = 输出总预算 - prefill 部分

    if len(str(decode_tokens)) > n_fmt_align:  # decode 计数超出 5 位
        n_fmt_align = len(str(decode_tokens))  # 按实际位数扩宽
        BAR_FORMAT_0 = BAR_FORMAT_0.replace("{n_fmt:>5}", "{n_fmt:>" + str(n_fmt_align) + "}")  # 更新格式串对齐宽度
        BAR_FORMAT_1 = BAR_FORMAT_0  # 同步给 decode 条格式

    if len(str(prefill_tokens)) < len(str(decode_tokens)):  # prefill 计数位数少于 decode 时
        old_align_str = "{n_fmt:>" + str(n_fmt_align) + "}"  # 记录当前格式串片段以便替换
        n_fmt_align += len(str(decode_tokens)) - len(str(prefill_tokens))  # 继续扩宽，使两条 token 进度条计数列对齐
        BAR_FORMAT_0 = BAR_FORMAT_0.replace(old_align_str, "{n_fmt:>" + str(n_fmt_align) + "}")  # 应用新宽度（BAR_FORMAT_1 保持不变）

    disabled = not use_pbar  # use_pbar=False 时禁用所有进度条
    input_pbar = tqdm(  # 创建"请求已发送"进度条
        total=num_requests,  # 总量 = 请求数
        desc="Requests sent",  # 描述文字
        position=0,  # 固定显示在第 0 行
        bar_format=BAR_FORMAT_0,  # 使用调整后的格式
        disable=disabled,  # 可整体禁用
    )
    output_pbar = tqdm(  # "请求已完成"进度条
        total=num_requests,  # 总量 = 请求数
        desc="Requests done",  # 描述
        position=1,  # 第 1 行
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    prefill_pbar = tqdm(  # "prefill token"进度条
        total=prefill_tokens,  # 总量 = 请求数（每请求 1 个首 token）
        desc="Prefill token",  # 描述
        position=2,  # 第 2 行
        bar_format=BAR_FORMAT_0,
        disable=disabled,
    )
    decode_pbar = tqdm(  # "decode token"进度条
        total=decode_tokens,  # 总量 = decode token 预算
        desc="Decode token ",  # 描述（尾部补空格与其他描述对齐）
        position=3,  # 第 3 行
        bar_format=BAR_FORMAT_1,  # 使用第二种格式
        disable=disabled,
    )
    return Console(  # 组装成 Console 返回
        input_pbar=input_pbar,
        output_pbar=output_pbar,
        prefill_pbar=prefill_pbar,
        decode_pbar=decode_pbar,
        disabled=disabled,
    )


def generate_prompt(tokenizer: Any, n: int) -> str:
    """Generate a prompt of approximately `n` tokens using the provided tokenizer.  用给定分词器生成恰好约 n 个 token 的随机 prompt 文本。"""
    vocab_size = tokenizer.vocab_size // 2  # 只取词表前半区，避开特殊 token 与罕见词，保证编解码往返稳定
    token_ids = [random.randint(0, vocab_size) for _ in range(n)]  # 随机采样 n 个 token id 作为初始猜测

    for _ in range(64):  # 最多迭代 64 轮，逼近"文本↔token 数"的不动点（decode/encode 往返会改变 token 数）
        prompt = tokenizer.decode(token_ids)  # ids → 文本（相邻 token 可能合并，长度会变化）
        token_ids = tokenizer.encode(prompt, add_special_tokens=False)  # 文本 → ids（往返后数量可能偏离 n）
        if len(token_ids) == n:  # 恰好等于目标长度
            return prompt  # 返回该文本（唯一成功出口）
        if len(token_ids) < n:  # 编码后变短了
            need = n - len(token_ids)  # 计算差额
            token_ids.extend([random.randint(0, vocab_size) for _ in range(need)])  # 随机补齐差额
        else:  # 编码后变长了
            token_ids = token_ids[:n]  # 直接截断到 n 个

    raise ValueError("Failed to generate a message of the desired length.")  # 64 轮仍未收敛则报错（理论上极少发生）


async def benchmark_one(
    client: OpenAI,  # OpenAI 异步客户端
    prompt: str,  # 请求 prompt 文本
    output_length: int,  # 期望输出 token 数
    model: str,  # 模型名
    *,
    pbar: Console | bool = True,  # 传入现成 Console 共享计数，或 bool 控制是否自建单请求进度条
    extra_body: Dict[str, Any] | None = None,  # 额外请求体参数（可覆盖下方默认 kwargs）
    input_length: int | None = None,  # a hack to force input length  # 私有扩展：强制声明输入长度（跳过服务端按实际内容统计）
) -> RawResult:
    """发送单个流式 chat 请求，记录每个 chunk 到达时间戳，返回原始结果。"""
    if isinstance(pbar, bool):  # 传布尔值时自建一个单请求控制台
        pbar = make_console(1, output_length, use_pbar=pbar)
    with pbar.inflight(1):  # 进入请求生命周期上下文（自动完成 发送→完成 计数）
        kwargs = {  # 构造压测专用请求参数
            "ignore_eos": True,  # 忽略 EOS，保证输出长度恰好等于 max_tokens（压测需要可控长度）
            "top_k": 1,  # 贪心解码（top_k=1），输出确定
        }
        # this is an internal kwargs that might work for our system  # 仅本系统支持的内部分段参数
        if input_length is not None:  # 需要强制输入长度时
            kwargs["input_length_override"] = input_length  # 传入覆盖参数
        kwargs.update(extra_body or {})  # 合并调用方参数，允许覆盖默认 kwargs（None 回退为空字典）
        response = await client.chat.completions.create(  # 发起流式 chat completion 请求
            model=model,  # 模型名
            stream=True,  # 流式返回，才能记录 TTFT 与逐 token 时间
            messages=[  # 单条 user 消息
                {
                    "role": "user",  # 用户角色
                    "content": prompt,  # prompt 内容
                },
            ],
            max_tokens=output_length,  # 最大输出 token 数
            temperature=0.0,  # 温度 0：确定性输出，压测结果可复现
            extra_body=kwargs,  # 压测附加参数（经 extra_body 透传给服务端）
        )
        tics = [time.perf_counter()]  # 记录请求发出时刻（首元素；后续元素为各 chunk 到达时刻）
        async for _ in response:  # 逐个消费流式 chunk
            tics.append(time.perf_counter())  # 每收到一个 chunk 记一次时间戳
            if len(tics) == 2:  # 收到第 1 个 chunk（tics 成为第 2 个元素）
                pbar.update_prefill()  # 标记该请求 prefill 完成（即 TTFT 时刻）
            elif len(tics) <= output_length + 1:  # 第 2..output_length 个 chunk 对应 decode token
                pbar.update_decode()  # 累加 decode token 计数
        return RawResult(  # 返回该请求的原始记录
            input_len=input_length,  # 输入长度（可能为 None，事后补）
            output_len=output_length,  # 输出长度
            message=prompt,  # prompt 原文
            tics=tics,  # 时间戳序列
        )


async def benchmark_one_batch(
    client: OpenAI,  # OpenAI 异步客户端
    prompts: List[str],  # 一批 prompt 文本
    output_lengths: List[int] | int,  # 每条的输出长度（或统一一个值）
    model: str,  # 模型名
    *,
    extra_body: Dict[str, Any] | None = None,  # 附加请求参数
    input_lengths: List[int | None] | None = None,  # 每条的强制输入长度（可选）
    pbar: Console | bool = True,  # 控制台对象或开关
) -> List[RawResult]:
    """并发压测一批请求：所有请求同时发出，共享同一组进度条计数。"""
    if isinstance(output_lengths, int):  # 传入单一 int 时
        output_lengths = [output_lengths] * len(prompts)  # 广播为每条请求相同长度
    if isinstance(pbar, bool):  # 传布尔值时自建控制台
        pbar = make_console(len(prompts), sum(output_lengths), use_pbar=pbar)  # 输出总预算 = 各长度之和
    if input_lengths is None:  # 未提供输入长度覆盖时
        l: List[int | None] = [None] * len(prompts)  # 每条都用 None（不覆盖）
        input_lengths = l  # work-around for typing bug  # 先赋给中间变量规避类型检查器的推断问题

    tasks = [  # 为每条请求创建异步任务
        benchmark_one(
            client=client,  # 共享客户端
            prompt=prompt,  # 该条的 prompt
            output_length=output_length,  # 该条的输出长度
            model=model,  # 模型名
            pbar=pbar,  # 共享控制台（进度条计数累加）
            extra_body=extra_body,  # 附加参数
            input_length=input_length,  # 该条的输入长度覆盖
        )
        for prompt, output_length, input_length in zip(
            prompts, output_lengths, input_lengths, strict=True  # strict=True：三个列表长度不一致时立即报错
        )
    ]
    with pbar.log_stats():  # 批次结束后自动关闭进度条并打印峰值统计
        return await asyncio.gather(*tasks)  # 并发等待全部请求完成，按输入顺序返回结果


async def benchmark_trace(
    client: OpenAI,  # OpenAI 异步客户端
    msgs: List[BenchmarkTrace],  # trace 请求列表（含到达时间戳）
    model: str,  # 模型名
    *,
    pbar: Console | bool = True,  # 控制台对象或开关
) -> List[RawResult]:
    """按 trace 中的时间戳回放请求，模拟真实负载的到达节奏。"""
    if isinstance(pbar, bool):  # 传布尔值时自建控制台
        sum_output_len = sum(msg.output_length for msg in msgs)  # 输出 token 总预算
        pbar = make_console(len(msgs), sum_output_len, use_pbar=pbar)
    start = time.perf_counter()  # 回放起始时刻（统一时间基准）
    offset = min(msg.timestamp for msg in msgs) - 1  # 把最早的请求平移到 start+1s，留出调度余量

    async def benchmark_timed(msg: BenchmarkTrace):  # 单条按时间触发的压测协程
        target = start + msg.timestamp - offset  # 该请求的绝对目标发送时刻
        await asyncio.sleep(max(0, target - time.perf_counter()))  # 睡到目标时刻（已过点则立即发送，max 防负数）
        return await benchmark_one(  # 到点后照常发送请求
            client, msg.message, msg.output_length, model, pbar=pbar, input_length=msg.input_length
        )

    tasks = [benchmark_timed(msg) for msg in msgs]  # 为每条 trace 建立定时任务
    with pbar.log_stats():  # 结束后打印统计
        return await asyncio.gather(*tasks)  # 并发回放全部请求


@overload  # 重载签名 1：提供 tokenizer 时返回 BenchmarkResult
def process_benchmark_results(raw_data: List[RawResult], tokenizer: Any) -> BenchmarkResult: ...


@overload  # 重载签名 2：不提供 tokenizer 时返回 None
def process_benchmark_results(raw_data: List[RawResult]) -> None: ...


def process_benchmark_results(
    raw_data: List[RawResult],  # 全部请求的原始结果
    tokenizer: Any = UNSET,  # 可选分词器；提供时额外返回可序列化的 BenchmarkResult（UNSET 哨兵区分未传与 None）
) -> BenchmarkResult | None:
    """统计并打印 TTFT/TPOT/E2E/吞吐等指标；传入 tokenizer 时返回结构化结果。"""
    accum_times: List[float] = []  # 所有 decode 间隔（每 token 生成耗时）的汇总池
    first_times: List[float] = []  # 每请求的首 token 延迟（TTFT）池
    results = [r.tics for r in raw_data]  # 抽出全部时间戳序列
    for tics in results:  # 逐请求处理
        deltas: List[float] = []  # 该请求相邻时间戳的差值序列
        for i in range(len(tics) - 1):  # 相邻两两求差
            diff = tics[i + 1] - tics[i]  # 第 i 个间隔时长
            deltas.append(diff)  # 收集
        first_times.append(deltas[0])  # 第一个间隔 = 发出请求到收到首 token = TTFT
        accum_times.extend(deltas[1:])  # 其余间隔 = 每 token 解码耗时，并入总池

    e2e_times = [tics[-1] - tics[0] for tics in results]  # 每请求端到端耗时 = 末时间戳 - 发出时刻
    first_times.sort()  # 升序排序，便于按下标取分位数
    accum_times.sort()  # 同上
    e2e_times.sort()  # 同上

    def _print_stats(times: List[float], scale: float = 1.0) -> Tuple[float, ...]:
        """计算一组时间值的 avg/p50/p90/p99/max（scale 用于单位换算，如秒→毫秒）。"""
        assert len(times) > 0  # 空列表无法统计，断言拦截
        return (
            scale * sum(times) / len(times),  # avg  # 平均值
            scale * times[int(len(times) * 0.5)],  # p50  # 50 分位（取整下标近似）
            scale * times[int(len(times) * 0.9)],  # p90  # 90 分位
            scale * times[int(len(times) * 0.99)],  # p99  # 99 分位
            scale * max(times),  # max  # 最大值
        )

    def _fmt(x: float) -> str:
        """按数值大小自适应格式化：大数取整、中数 2 位小数、小数 4 位小数（统一右对齐 6 宽）。"""
        if x >= 1000:  # ≥1000
            return f"{int(x):>6}"  # 显示为整数
        elif x >= 10:  # [10, 1000)
            return f"{x:>6.2f}"  # 保留 2 位小数
        else:  # <10
            return f"{x:>6.4f}"  # 保留 4 位小数

    avg_ttft, p50_ttft, p90_ttft, p99_ttft, max_ttft = _print_stats(first_times, 1000)  # TTFT 统计（秒→毫秒）
    avg_tpot, p50_tpot, p90_tpot, p99_tpot, max_tpot = _print_stats(accum_times, 1000)  # TPOT 统计（秒→毫秒）
    avg_e2e, p50_e2e, p90_e2e, p99_e2e, max_e2e = _print_stats(e2e_times)  # E2E 统计（保持秒）

    min_time = min(min(r) for r in results)  # 全部时间戳的最小值（最早时刻）
    max_time = max(max(r) for r in results)  # 全部时间戳的最大值（最晚时刻）
    dur = max_time - min_time  # 整个压测的墙钟时长
    assert dur > 0, "Duration must be positive"  # 时长必须为正，否则数据异常

    num_tokens = sum(len(tic) for tic in results)  # token 事件总数（≈ 请求数 + 输出 token 数）
    num_requests = len(results)  # 请求总数

    logger.info(f"Num requests: #{num_requests}, Num tokens: #{num_tokens}")  # 打印请求与 token 规模
    logger.info(
        f"TTFT: {_fmt(avg_ttft)} ms (p50: {_fmt(p50_ttft)} ms, p90: {_fmt(p90_ttft)} ms,"
        f" p99: {_fmt(p99_ttft)} ms, max: {_fmt(max_ttft)} ms)"  # 打印首 token 延迟分布
    )
    logger.info(
        f"TPOT: {_fmt(avg_tpot)} ms (p50: {_fmt(p50_tpot)} ms, p90: {_fmt(p90_tpot)} ms,"
        f" p99: {_fmt(p99_tpot)} ms, max: {_fmt(max_tpot)} ms)"  # 打印每 token 解码延迟分布
    )
    logger.info(
        f"E2E:  {_fmt(avg_e2e) }  s (p50: {_fmt(p50_e2e) }  s, p90: {_fmt(p90_e2e) }  s,"
        f" p99: {_fmt(p99_e2e) }  s, max: {_fmt(max_e2e) }  s)"  # 打印端到端耗时分布
    )
    logger.info(f"Duration: {_fmt(dur)} s")  # 打印压测总时长
    logger.info(f"Throughput: {_fmt(num_tokens / dur)} token/s, {_fmt(num_requests / dur)} req/s")  # 打印 token 吞吐与请求吞吐

    # normalize the time to start from zero  # 把时间戳平移为从 0 开始
    results = [[r - min_time for r in tics] for tics in results]  # 归一化时间基准（便于后续可视化/保存）
    if isinstance(tokenizer, Unset):  # 未提供 tokenizer（保持 UNSET 哨兵原样）
        return None  # 只打印统计，不返回结构化结果

    return BenchmarkResult(  # 组装可序列化的最终结果
        raw_data=[
            BenchOneResult(
                tics=r.tics,  # 原始时间戳
                input_len=(
                    r.input_len  # 已知输入长度直接使用
                    if r.input_len is not None
                    else len(tokenizer.encode(r.message, add_special_tokens=False))  # 未知则现场用分词器统计 prompt 长度
                ),
                output_len=r.output_len,  # 输出长度
            )
            for r in raw_data  # 逐请求转换
        ]
    )


def read_qwen_trace(
    file_path: str,  # trace jsonl 文件路径
    tokenizer: Any,  # 分词器（dummy 模式下用于构造伪 prompt）
    n: int | None = None,  # 只读前 n 条（None 表示读全部）
    dummy: bool = False,  # True：共享前缀截断构造输入（快）；False：逐条随机生成（慢但相互独立）
) -> List[BenchmarkTrace]:
    """读取 Qwen 百炼 trace（jsonl 格式），转换为压测请求列表。"""

    class JSONInput(BaseModel):  # pydantic 模型：定义 trace 每行的字段
        chat_id: int  # 会话 id
        parent_chat_id: int  # 父会话 id
        timestamp: float  # 到达时间戳（秒）
        input_length: int  # 输入 token 长度
        output_length: int  # 输出 token 长度
        type: str  # unused  # 未使用字段
        turn: int  # unused  # 未使用字段
        hash_ids: List[int]  # unused  # 未使用字段（前缀块 hash，可用于前缀缓存复用研究）

    with open(file_path, "r") as f:  # 打开 trace 文件
        lines = f.readlines()  # 一次性读入全部行
        if n is not None:  # 限制条数时
            lines = lines[:n]  # 只保留前 n 行
    objs = [JSONInput.model_validate_json(line) for line in lines]  # 逐行 JSON 校验并转为对象
    if dummy:  # dummy 模式：全部请求共享一个长 prompt 的前缀
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))  # 按最大输入长度生成一份随机文本
        ids = tokenizer.encode(prompt, add_special_tokens=False)  # 编码成 ids 备用
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])  # 每条请求取前缀再解码（省去逐条随机生成，速度快）
    else:  # 非 dummy：逐条独立生成随机 prompt
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)  # 每条按自身长度独立生成
    return [
        BenchmarkTrace(  # 转成压测请求记录
            timestamp=obj.timestamp,  # 到达时间
            message=_get_prompt(obj),  # 构造出的 prompt
            input_length=obj.input_length,  # 输入长度
            output_length=obj.output_length,  # 输出长度
        )
        for obj in objs  # 遍历全部 trace 条目
    ]


def read_mooncake_trace(
    file_path: str,  # Mooncake trace 文件路径
    tokenizer: Any,  # 分词器
    n: int | None = None,  # 只读前 n 条
    dummy: bool = False,  # 同 read_qwen_trace 的 dummy 语义
) -> List[BenchmarkTrace]:
    """读取 Mooncake trace（jsonl 格式），逻辑与 Qwen trace 类似，但时间戳单位为毫秒。"""

    class JSONInput(BaseModel):  # pydantic 模型：Mooncake trace 字段更少
        timestamp: int  # 到达时间戳（毫秒，注意与 Qwen trace 的秒不同）
        input_length: int  # 输入长度
        output_length: int  # 输出长度
        hash_ids: List[int]  # unused for now  # 暂未使用

    with open(file_path, "r") as f:  # 打开文件
        lines = f.readlines()  # 读全部行
        if n is not None:  # 限制条数
            lines = lines[:n]  # 截取前 n 行
    objs = [JSONInput.model_validate_json(line) for line in lines]  # 校验解析
    if dummy:  # dummy 模式（与 Qwen 相同的策略）
        prompt = generate_prompt(tokenizer, max(obj.input_length for obj in objs))  # 生成最长的一份随机文本
        ids = tokenizer.encode(prompt, add_special_tokens=False)  # 编码备用
        _get_prompt = lambda obj: tokenizer.decode(ids[: obj.input_length])  # 前缀截断构造
    else:  # 非 dummy
        _get_prompt = lambda obj: generate_prompt(tokenizer, obj.input_length)  # 按长度逐条独立生成
    return [
        BenchmarkTrace(
            timestamp=obj.timestamp / 1000,  # 毫秒 → 秒（与 Qwen trace 统一单位）
            message=_get_prompt(obj),  # 构造出的 prompt
            input_length=obj.input_length,  # 输入长度
            output_length=obj.output_length,  # 输出长度
        )
        for obj in objs  # 遍历全部条目
    ]


def scale_traces(
    traces: List[BenchmarkTrace],  # 原始 trace 列表
    scale: float,  # 时间缩放因子：<1 压缩到达间隔（更高负载），>1 拉长间隔（更低负载）
) -> List[BenchmarkTrace]:
    """把 trace 时间戳归零后按 scale 缩放，并按新时间戳排序返回。"""
    min_tic = min(trace.timestamp for trace in traces)  # 最早到达时间（用于归零）
    return sorted(
        [
            BenchmarkTrace(
                timestamp=(trace.timestamp - min_tic) * scale,  # 归零后按 scale 缩放到达间隔
                message=trace.message,  # prompt 保持不变
                input_length=trace.input_length,  # 输入长度不变
                output_length=trace.output_length,  # 输出长度不变
            )
            for trace in traces  # 逐条变换
        ],
        key=lambda x: x.timestamp,  # 按新时间戳升序排序，保证回放顺序正确
    )


async def get_model_name(client: OpenAI) -> str:
    """查询服务端可用模型列表并返回第一个模型名。"""
    async for model in client.models.list():  # 异步遍历模型列表
        return model.id  # 取第一个模型 id 即可
    raise ValueError("No models available")  # 列表为空时抛错
