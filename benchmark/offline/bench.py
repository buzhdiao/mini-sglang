# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py

# time 用于统计整个批量生成阶段的耗时
import time
# randint 生成随机长度，seed 固定随机种子保证可复现
from random import randint, seed

# 采样参数：温度、max_tokens、是否忽略 EOS 等
from minisgl.core import SamplingParams
# 离线批量推理的高层封装入口
from minisgl.llm import LLM


# 离线压测主流程：固定一批随机请求，测引擎端到端吞吐
def main():
    # 固定全局随机种子，保证每次跑出的请求长度分布一致
    seed(0)
    # 请求总数（批次大小），与 cuda_graph_max_bs 对齐以吃到 CUDA Graph
    num_seqs = 256
    # 单条请求输入 token 数的上限
    max_input_len = 1024
    # 单条请求输出 token 数的上限
    max_ouput_len = 1024

    # align the hyperparameters
    # 创建 LLM 实例并按压测场景对齐超参数：输入+输出需落在 4096 上下文内；
    # max_extend_tokens 控制 chunked prefill 单批预算；cuda_graph_max_bs=256 让 decode 走 CUDA Graph；page_size 是 KV cache 分页大小
    llm = LLM(
        "Qwen/Qwen3-0.6B",
        max_seq_len_override=4096,
        max_extend_tokens=16384,
        cuda_graph_max_bs=256,
        page_size=256,
    )

    # 构造 256 条请求的输入：直接给 token id，绕过分词器；每条长度在 [100,1024) 随机
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    # 每条请求各自的采样参数：ignore_eos=True 强制生成满 max_tokens，使输出长度可控、跑分稳定
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    # 先发一条极短请求做预热，触发 flashinfer 等内核的初始化/编译，避免计入正式压测耗时
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer
    # 记录正式压测开始时间
    t = time.time()
    # 一次性提交全部 256 条请求，由引擎内部 continuous batching 调度
    llm.generate(prompt_token_ids, sampling_params)
    # 计算总耗时
    t = time.time() - t
    # 理论输出 token 总数（ignore_eos 下实际输出即等于该预算）
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    # 端到端吞吐 = 总 token 数 / 总耗时
    throughput = total_tokens / t
    # 打印吞吐结果
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


# 仅在直接运行该脚本时执行压测
if __name__ == "__main__":
    # 入口
    main()
