# Adapted from: https://github.com/GeeeekExplorer/nano-vllm/blob/main/bench.py  # 改编自 nano-vllm 的离线压测脚本

import time  # 导入 time 模块，用于统计推理总耗时
from random import randint, seed  # 导入随机整数生成与随机种子设置，用于构造随机压测输入并保证可复现

from minisgl.core import SamplingParams  # 导入采样参数配置类（温度、最大生成长度等）
from minisgl.llm import LLM  # 导入 LLM 高层封装类，负责模型加载与离线批量推理


def main():
    seed(0)  # 固定随机种子，保证每次运行生成的输入长度/内容一致，便于公平对比性能
    num_seqs = 256  # 压测的请求数量（序列条数）
    max_input_len = 1024  # 单条输入 prompt 的最大 token 长度
    max_ouput_len = 1024  # 单条输出的最大生成 token 长度（注意：原变量名拼写为 ouput，保持原样不改）

    # align the hyperparameters  # 与参考实现对齐的超参数设置
    llm = LLM(
        "Qwen/Qwen3-0.6B",  # 加载 Qwen3-0.6B 小模型，便于快速压测
        max_seq_len_override=4096,  # 覆盖模型默认最大序列长度为 4096（需容纳 输入+输出）
        max_extend_tokens=16384,  # 单轮 prefill 最多扩展的 token 数（决定一次能吃下多少 prefill 计算）
        cuda_graph_max_bs=256,  # CUDA Graph 支持的最大 batch size，与压测请求数 256 匹配
        page_size=256,  # KV Cache 分页大小：每页 256 个 token
    )

    # 构造 256 条随机 token 序列作为输入 prompt（每条长度在 100~1024 之间随机，id 范围 [0,10000)）
    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)
    ]
    # 为每条输入配置采样参数：温度 0.6、忽略 EOS（保证生成长度可预测）、随机最大生成长度 100~1024
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len))
        for _ in range(num_seqs)
    ]
    llm.generate(["Benchmark: "], SamplingParams(temperature=0.1))  # to warm up flashinfer  # 先跑一次小推理预热 flashinfer 等 CUDA kernel，避免首次 JIT/加载开销污染计时
    t = time.time()  # 记录压测开始时刻
    llm.generate(prompt_token_ids, sampling_params)  # 正式执行离线批量推理（引擎内部持续调度直到全部请求完成）
    t = time.time() - t  # 计算压测总耗时（秒）
    total_tokens = sum(sp.max_tokens for sp in sampling_params)  # 理论生成 token 总数（ignore_eos=True 时实际生成长度≈max_tokens 之和）
    throughput = total_tokens / t  # 吞吐量 = 总 token 数 / 总时间
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")  # 打印压测汇总结果


if __name__ == "__main__":  # 作为脚本直接运行时执行 main
    main()
