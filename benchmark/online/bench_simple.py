import asyncio  # 导入 asyncio，支持异步并发压测与事件循环
import random  # 导入 random，生成随机输入/输出长度并固定种子
import sys  # 导入 sys，出错时以非零退出码结束进程
from typing import List  # 导入 List 类型注解

from minisgl.benchmark.client import (  # 导入压测客户端工具函数
    benchmark_one,  # 发送单个请求并记录各阶段时间戳
    benchmark_one_batch,  # 并发压测一批请求
    generate_prompt,  # 生成指定 token 长度的随机 prompt
    get_model_name,  # 从服务端查询模型名
    process_benchmark_results,  # 汇总打印压测指标
)
from minisgl.utils import init_logger  # 导入日志初始化工具
from openai import AsyncOpenAI as OpenAI  # 导入 OpenAI 异步客户端
from transformers import AutoTokenizer  # 导入分词器，用于构造 prompt

logger = init_logger(__name__)  # 创建本模块 logger


async def main():
    """简单在线压测：连通性检查 → 生成随机长度消息 → 按档位批量压测并输出指标。"""
    try:  # 顶层异常兜底，保证出错时打印信息并以非零码退出
        random.seed(42)  # reproducibility  # 固定随机种子保证可复现

        async def generate_task(max_bs: int) -> List[str]:
            """Generate a list of tasks with random lengths.  生成 max_bs 条随机长度的 prompt 文本。"""
            result = []  # 存放生成的消息列表
            for _ in range(max_bs):  # 逐条生成
                length = random.randint(1, MAX_INPUT)  # 随机长度，范围 [1, MAX_INPUT]
                message = generate_prompt(tokenizer, length)  # 生成恰好该 token 长度的随机 prompt
                result.append(message)  # 收集到结果列表
                await asyncio.sleep(0)  # 主动让出事件循环，避免长时间占用协程调度
            return result  # 返回全部消息

        TEST_BS = [64]  # 待测试的 batch size 档位列表
        PORT = 1919  # 本地推理服务监听端口
        MAX_INPUT = 8192  # 输入 prompt 的最大 token 长度
        # Create the async client  # 创建异步客户端
        async with OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key="dummy") as client:  # 连接本地服务，退出时自动关闭连接
            MODEL = await get_model_name(client)  # 查询服务端实际部署的模型名
            tokenizer = AutoTokenizer.from_pretrained(MODEL)  # 加载对应分词器

            logger.info(f"Loaded tokenizer from {MODEL}")  # 记录分词器加载完成
            logger.info("Testing connection to server...")  # 提示即将测试服务连通性

            # Test connection with a simple request first  # 先用一个小请求验证连通性
            try:  # 连接测试阶段的异常单独处理
                gen_task = asyncio.create_task(generate_task(max(TEST_BS)))  # 后台并发预生成全部大 prompt（与连通性测试重叠执行以省时间）
                test_msg = generate_prompt(tokenizer, 100)  # 构造 100 token 的测试消息
                test_result = await benchmark_one(client, test_msg, 2, MODEL, pbar=False)  # 发送小请求（最多 2 个输出 token）验证服务可用
                if len(test_result.tics) <= 2:  # 时间戳不足 3 个说明没收到任何流式 chunk（prefill 未成功）
                    logger.info("Server connection test failed")  # 记录失败
                    return  # 直接退出
                logger.info("Server connection successful")  # 记录连通成功
            except Exception as e:  # 连不上等异常
                logger.warning("Server connection failed")  # 记录失败警告
                logger.warning(f"Make sure the server is running on http://127.0.0.1:{PORT}")  # 提示先启动推理服务
                raise e from e  # 重新抛出，交由外层统一处理

            msgs = await gen_task  # 等待后台生成的全部测试消息
            output_lengths = [random.randint(16, 1024) for _ in range(max(TEST_BS))]  # 为每条消息随机指定输出长度（16~1024 token）
            logger.info(f"Generated {len(msgs)} test messages")  # 记录消息生成完成

            logger.info("Running benchmark...")  # 提示正式压测开始
            for batch_size in TEST_BS:  # 逐档 batch size 压测
                try:  # 单档失败不影响其他档位
                    results = await benchmark_one_batch(
                        client, msgs[:batch_size], output_lengths[:batch_size], MODEL  # 截取前 batch_size 条消息及其输出长度，并发压测
                    )
                    process_benchmark_results(results)  # 汇总打印该档位指标
                except Exception as e:  # 该档位出错
                    logger.info(f"Error with batch size {batch_size}: {e}")  # 记录错误并继续下一档
                    continue
            logger.info("Benchmark completed.")  # 记录压测完成

    except Exception as e:  # 顶层兜底捕获
        print(f"Error in main: {e}")  # 打印错误信息
        sys.exit(1)  # 以非零退出码结束，便于脚本化检测失败


if __name__ == "__main__":  # 作为脚本直接运行时启动事件循环
    asyncio.run(main())
