from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许在注解中使用前向引用类型

import asyncio  # 导入 asyncio，用于运行异步压测主流程
import os  # 导入 os，获取当前文件路径信息
import random  # 导入 random，固定随机种子保证可复现
from pathlib import Path  # 导入 Path，构造 trace 文件的本地保存路径

from minisgl.benchmark.client import (  # 导入压测客户端工具函数
    benchmark_trace,  # 按 trace 时间戳回放请求的压测函数
    get_model_name,  # 从服务端查询当前部署的模型名
    process_benchmark_results,  # 汇总并打印 TTFT/TPOT/E2E 等指标
    read_qwen_trace,  # 读取并解析 Qwen 真实负载 trace
    scale_traces,  # 对 trace 时间戳做缩放，调节请求到达速率
)
from minisgl.utils import init_logger  # 导入日志初始化工具
from openai import AsyncOpenAI as OpenAI  # 导入 OpenAI 异步客户端，通过 OpenAI 兼容 API 访问推理服务
from transformers import AutoTokenizer  # 导入分词器，按 trace 记录的长度构造 prompt

logger = init_logger(__name__)  # 创建本模块的 logger

URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"  # Qwen 百炼真实使用 trace（场景 A，16 token 分块）的下载地址


def download_qwen_trace(url: str) -> str:
    """下载 Qwen trace 文件到脚本所在目录（已存在则跳过），返回本地文件路径。"""
    dir = Path(os.path.dirname(__file__))  # 当前脚本所在目录
    # download the file if not exists  # 文件不存在时才下载
    file_path = dir / "qwen_traceA_blksz_16.jsonl"  # 本地保存路径（与脚本同目录）
    if not file_path.exists():  # 本地不存在才执行下载
        import urllib.request  # 函数内延迟导入：不下载时无需加载该模块

        logger.info(f"Downloading trace from {url} to {file_path}...")  # 记录下载开始日志
        urllib.request.urlretrieve(url, file_path)  # 下载文件到本地路径
        logger.info("Download completed.")  # 记录下载完成日志
    return str(file_path)  # 返回本地文件路径字符串


async def main():
    """在线压测主流程：读取真实 trace → 多档速率缩放回放 → 输出各档位性能指标。"""
    random.seed(42)  # reproducibility  # 固定随机种子，保证每次实验可复现
    PORT = 1919  # 本地推理服务监听端口
    N = 1000  # 使用 trace 中前 1000 条请求
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow  # 时间缩放因子：越小请求到达越密集（负载越重），1.6 最慢
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:  # 连接本地 OpenAI 兼容服务，退出时自动关闭
        MODEL = await get_model_name(client)  # 从服务端获取实际部署的模型名
        tokenizer = AutoTokenizer.from_pretrained(MODEL)  # 加载对应分词器，用于按长度构造 prompt
        TRACES = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=N, dummy=True)  # 下载并解析前 N 条 trace；dummy=True 表示用共享长前缀截断构造伪文本（快）
        logger.info(f"Start benchmarking with {N} requests using model {MODEL}...")  # 记录压测开始日志
        for scale in SCALES:  # 逐档缩放系数压测
            traces = scale_traces(TRACES, scale)  # 时间戳归零后按 scale 缩放（控制到达间隔）
            results = await benchmark_trace(client, traces, MODEL)  # 按时间戳回放整批请求并收集结果
            process_benchmark_results(results)  # 汇总打印该档位的 TTFT/TPOT/吞吐等指标
        logger.info("Benchmarking completed.")  # 记录压测完成日志


if __name__ == "__main__":  # 作为脚本直接运行时启动事件循环
    asyncio.run(main())
