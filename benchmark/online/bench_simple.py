# asyncio 驱动异步并发压测
import asyncio
# random 生成随机输入/输出长度
import random
# sys 在出错时以非零码退出
import sys
# List 类型注解
from typing import List

# 复用基准测试客户端的各工具函数
from minisgl.benchmark.client import (
    benchmark_one,
    benchmark_one_batch,
    generate_prompt,
    get_model_name,
    process_benchmark_results,
)
# 统一日志初始化
from minisgl.utils import init_logger
# OpenAI 兼容异步客户端
from openai import AsyncOpenAI as OpenAI
# AutoTokenizer 加载模型分词器
from transformers import AutoTokenizer

# 创建模块级 logger
logger = init_logger(__name__)


# 简单在线压测：构造一批随机长度请求，并发打到本地推理服务
async def main():
    # 整体兜底异常，保证出错时打印信息并以非零码退出
    try:
        # 固定随机种子保证可复现
        random.seed(42)  # reproducibility

        # 异步生成 max_bs 条随机长度的测试消息
        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of tasks with random lengths."""
            # 存放生成的消息
            result = []
            # 逐条生成
            for _ in range(max_bs):
                # 输入 token 长度在 [1, MAX_INPUT] 内随机
                length = random.randint(1, MAX_INPUT)
                # 造出指定 token 长度的一段文本
                message = generate_prompt(tokenizer, length)
                # 收集消息
                result.append(message)
                # 主动让出事件循环，避免构造阶段饿死其他协程
                await asyncio.sleep(0)
            # 返回全部测试消息
            return result

        # 要测试的批量大小列表
        TEST_BS = [64]
        # 本地推理服务端口
        PORT = 1919
        # 输入 token 长度上限
        MAX_INPUT = 8192
        # Create the async client
        # 建立 OpenAI 兼容异步客户端会话（with 结束时自动关闭连接）
        async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:
            # 从服务端查询模型名，避免硬编码
            MODEL = await get_model_name(client)
            # 加载对应模型的分词器
            tokenizer = AutoTokenizer.from_pretrained(MODEL)

            # 记录分词器加载完成
            logger.info(f"Loaded tokenizer from {MODEL}")
            # 提示即将测试服务连通性
            logger.info("Testing connection to server...")

            # Test connection with a simple request first
            # 先用一条极小请求验证服务可用，失败则直接终止
            try:
                # 后台并发构造全部测试消息（与连通性测试并行，节省时间）
                gen_task = asyncio.create_task(generate_task(max(TEST_BS)))
                # 构造 100 token 的探活消息
                test_msg = generate_prompt(tokenizer, 100)
                # 发送探活请求，只要 2 个 token 输出，关闭进度条
                test_result = await benchmark_one(client, test_msg, 2, MODEL, pbar=False)
                # 时间戳过少说明请求未真正完成（服务异常）
                if len(test_result.tics) <= 2:
                    # 判定连接测试失败
                    logger.info("Server connection test failed")
                    # 直接结束压测
                    return
                # 连通性 OK
                logger.info("Server connection successful")
            # 任何异常都视为服务未就绪
            except Exception as e:
                # 记录失败
                logger.warning("Server connection failed")
                # 提示先在对应端口启动服务
                logger.warning(f"Make sure the server is running on http://127.0.0.1:{PORT}")
                # 原样抛出，交给外层统一处理
                raise e from e

            # 等待后台消息构造完成
            msgs = await gen_task
            # 每条请求期望的输出长度（16~1024 随机），控制 decode 负载
            output_lengths = [random.randint(16, 1024) for _ in range(max(TEST_BS))]
            # 记录生成的消息条数
            logger.info(f"Generated {len(msgs)} test messages")

            # 提示正式压测开始
            logger.info("Running benchmark...")
            # 逐个批量大小执行压测
            for batch_size in TEST_BS:
                # 单个批量失败不影响后续批量
                try:
                    # 并发发送 batch_size 条请求并等待全部完成
                    results = await benchmark_one_batch(
                        client, msgs[:batch_size], output_lengths[:batch_size], MODEL
                    )
                    # 汇总打印该批量的吞吐/延迟指标
                    process_benchmark_results(results)
                # 捕获批量级异常
                except Exception as e:
                    # 记录错误后继续下一个批量
                    logger.info(f"Error with batch size {batch_size}: {e}")
                    continue
            # 压测结束
            logger.info("Benchmark completed.")

    # 顶层兜底
    except Exception as e:
        # 打印错误摘要
        print(f"Error in main: {e}")
        # 非零退出码表示失败
        sys.exit(1)


# 直接运行脚本时执行
if __name__ == "__main__":
    # 启动事件循环执行 async 主流程
    asyncio.run(main())
