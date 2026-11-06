# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# asyncio 驱动异步压测主流程
import asyncio
# os 用于取当前脚本所在目录
import os
# random 固定随机种子保证可复现
import random
# Path 做路径拼接
from pathlib import Path

# 复用基准测试客户端的各工具函数
from minisgl.benchmark.client import (
    benchmark_trace,
    get_model_name,
    process_benchmark_results,
    read_qwen_trace,
    scale_traces,
)
# 统一日志初始化
from minisgl.utils import init_logger
# OpenAI 兼容异步客户端，连接本地推理服务
from openai import AsyncOpenAI as OpenAI
# AutoTokenizer 加载模型分词器
from transformers import AutoTokenizer

# 创建模块级 logger
logger = init_logger(__name__)

# Qwen 真实线上请求轨迹（匿名化），含每条请求的到达时间与输入/输出长度
URL = "https://media.githubusercontent.com/media/alibaba-edu/qwen-bailian-usagetraces-anon/refs/heads/main/qwen_traceA_blksz_16.jsonl"


# 下载轨迹文件（若本地不存在），返回本地路径
def download_qwen_trace(url: str) -> str:
    # 轨迹文件存放在本脚本同目录
    dir = Path(os.path.dirname(__file__))
    # download the file if not exists
    # 目标文件路径
    file_path = dir / "qwen_traceA_blksz_16.jsonl"
    # 首次运行才下载，已存在则跳过
    if not file_path.exists():
        # 函数内延迟导入，避免模块加载即引入该依赖
        import urllib.request

        # 记录下载开始
        logger.info(f"Downloading trace from {url} to {file_path}...")
        # 阻塞式下载到目标路径
        urllib.request.urlretrieve(url, file_path)
        # 记录下载完成
        logger.info("Download completed.")
    # 返回本地文件路径字符串
    return str(file_path)


# 在线压测主流程：回放真实轨迹并扫描不同负载强度
async def main():
    # 固定随机种子，轨迹处理可复现
    random.seed(42)  # reproducibility
    # 本地推理服务端口
    PORT = 1919
    # 取轨迹前 1000 条请求
    N = 1000
    # 相邻请求到达间隔的缩放系数：越小请求越密集、负载越高；1.6 则放宽 60%
    SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 1.6]  # from fast to slow
    # 建立异步 OpenAI 客户端会话（with 结束时自动关闭连接）
    async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
        # 从服务端 /v1/models 查询模型名，避免硬编码
        MODEL = await get_model_name(client)
        # 加载对应模型的分词器
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        # 下载（若缺）并解析轨迹：dummy 模式用随机文本按长度近似真实输入
        TRACES = read_qwen_trace(download_qwen_trace(URL), tokenizer, n=N, dummy=True)
        # 记录压测开始
        logger.info(f"Start benchmarking with {N} requests using model {MODEL}...")
        # 逐个负载强度执行压测
        for scale in SCALES:
            # 缩放相邻请求的时间间隔，得到新的负载曲线
            traces = scale_traces(TRACES, scale)
            # 按轨迹时间戳回放全部请求并收集各请求时间点
            results = await benchmark_trace(client, traces, MODEL)
            # 汇总打印该强度下的吞吐/延迟等指标
            process_benchmark_results(results)
        # 全部压测结束
        logger.info("Benchmarking completed.")


# 直接运行脚本时执行
if __name__ == "__main__":
    # 启动事件循环执行 async 主流程
    asyncio.run(main())
