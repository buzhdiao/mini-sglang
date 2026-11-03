import shutil  # 导入 shutil，用于把下载的网络数据流式拷贝到本地文件
import time  # 导入 time，用于统计压测总耗时
import urllib.request  # 导入 urllib.request，用于从 HuggingFace 下载数据集
from pathlib import Path  # 导入 Path，用于跨平台路径拼接与文件存在性判断
from random import seed  # 导入 seed，固定随机种子保证实验可复现

import pyarrow.parquet as pq  # 导入 pyarrow 的 parquet 读取模块，流式读取 WildChat 数据集
from minisgl.core import SamplingParams  # 导入采样参数配置类
from minisgl.llm import LLM  # 导入 LLM 高层封装，负责模型加载与离线批量推理
from transformers import AutoTokenizer  # 导入 HuggingFace 分词器，把真实文本转成 token ids

LANGS = {"English", "Chinese"}  # 语言白名单：只保留英文/中文对话样本，过滤其他语言

BASE_DIR = Path(__file__).resolve().parent  # 当前脚本所在目录，作为数据集下载目录
WILDCHAT_FIRST_SHARD = "train-00000-of-00086.parquet"  # WildChat 数据集第一个分片的文件名
WILDCHAT_BASE_URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/"  # 数据集分片的下载基础 URL


def download_if_missing(url: str, path: Path) -> None:
    """若目标文件不存在则从 url 下载到 path；已存在则跳过（实现本地缓存，避免重复下载）。"""
    if path.exists():  # 文件已存在
        return  # 直接返回，跳过下载
    print(f"Downloading {url} -> {path}")  # 打印下载提示
    path.parent.mkdir(parents=True, exist_ok=True)  # 递归创建父目录（已存在不报错）
    with urllib.request.urlopen(url, timeout=300) as resp, path.open("wb") as out:  # 打开网络响应与本地写入文件（300 秒超时）
        shutil.copyfileobj(resp, out)  # 流式拷贝网络数据到本地，避免整个文件一次性载入内存


def iter_filtered_prompt_ids(tokenizer):
    """生成器：逐条产出经过滤的 WildChat 首轮用户提问的 token ids（惰性求值，内存友好）。"""
    shard_path = BASE_DIR / WILDCHAT_FIRST_SHARD  # 本地分片文件路径
    download_if_missing(WILDCHAT_BASE_URL + WILDCHAT_FIRST_SHARD, shard_path)  # 本地缺失则先下载

    parquet = pq.ParquetFile(shard_path)  # 打开 parquet 文件句柄
    for batch in parquet.iter_batches(batch_size=256, columns=["conversation"]):  # 每次只读 256 行的 conversation 列，流式遍历避免内存爆炸
        prompts = []  # 收集本批次通过过滤的文本
        for conv in batch.to_pydict()["conversation"]:  # 遍历本批次的每段对话
            if not conv:  # 空对话
                continue  # 直接跳过

            first_user = None  # 记录对话中第一条用户发言
            for turn in conv:  # 按顺序遍历对话轮次
                if turn.get("role") == "user":  # 找到第一条用户消息
                    first_user = turn  # 保存
                    break  # 找到即停止
            if first_user is None:  # 整段对话没有用户消息
                continue  # 跳过

            text = (first_user.get("content") or "").strip()  # 取出消息文本并去首尾空白（None 时回退为空串）
            if not text:  # 空文本
                continue  # 跳过
            if first_user.get("language") not in LANGS:  # 语言不在白名单（英/中）内
                continue  # 跳过
            if bool(first_user.get("redacted")) or bool(first_user.get("toxic")):  # 被打码（含隐私）或有毒样本
                continue  # 跳过
            prompts.append(text)  # 通过全部过滤条件，收集该文本

        if not prompts:  # 本批次无有效样本
            continue  # 继续处理下一批

        for prompt in prompts:  # 对每个有效文本套用聊天模板并转成 token ids
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],  # 构造单条 user 消息
                tokenize=True,  # 直接返回 token ids 而非模板字符串
                add_generation_prompt=True,  # 在末尾追加"assistant 开始生成"的模板片段
            )
            yield ids  # 以生成器方式逐条产出


def print_len_stats(name: str, lengths: list[int]) -> None:
    """打印一组长度值的统计信息：数量、min、p50、p90、p99、max。"""
    if not lengths:  # 空列表无法统计
        print(f"{name}: no data")  # 打印无数据提示
        return  # 提前返回
    arr = sorted(lengths)  # 升序排序，便于按下标取分位数
    n = len(arr)  # 样本总数
    print(
        f"{name}: count={n}, min={arr[0]}, p50={arr[int(0.50*n)]}, "  # 总数、最小值与 50 分位（取整下标取近似分位）
        f"p90={arr[int(0.90*n)]}, p99={arr[int(0.99*n)]}, max={arr[-1]}"  # 90/99 分位与最大值
    )


def main() -> None:
    """压测主流程：下载真实 WildChat 数据 → 过滤构造输入 → 预热 → 批量推理 → 统计吞吐与长度分布。"""
    seed(0)  # 固定随机种子保证可复现
    MODEL = "Qwen/Qwen3-0.6B"  # 被压测的模型
    NUM_SEQS = 256 * 1  # 压测请求总数
    MAX_OUTPUT_LEN = 4096  # 每条请求的最大生成长度
    IGNORE_EOS = False  # 不忽略 EOS：按模型自然停止，输出长度更接近真实分布

    tokenizer = AutoTokenizer.from_pretrained(MODEL)  # 加载模型对应的分词器

    prompt_token_ids = []  # 收集压测输入 token ids
    for ids in iter_filtered_prompt_ids(tokenizer):  # 逐条获取过滤后的真实样本
        prompt_token_ids.append(ids)  # 累积到列表
        if len(prompt_token_ids) >= NUM_SEQS:  # 凑够 NUM_SEQS 条即停止
            break  # 提前退出生成器

    num_reqs = len(prompt_token_ids)  # 实际拿到的请求数（数据不足时可能小于 NUM_SEQS）
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=IGNORE_EOS, max_tokens=MAX_OUTPUT_LEN)  # 所有请求统一采样参数
        for _ in range(num_reqs)  # 每条请求一份
    ]

    llm = LLM(MODEL)  # 初始化 LLM 引擎（加载权重、分配 KV Cache 等一次性开销）

    # warm up  # 预热：先用第一条请求跑一次，触发 JIT 编译/显存分配/缓存建立等一次性开销
    warmup_result = llm.generate(
        [prompt_token_ids[0]],  # 只喂一条输入
        sampling_params[0],  # 对应的采样参数
    )[0]  # 取出该条请求的生成结果字典
    templated_input_preview = tokenizer.decode(
        prompt_token_ids[0],  # 把模板化后的输入 ids 解码回文本，便于人工检查模板是否正确
        skip_special_tokens=False,  # 保留特殊 token，展示完整模板结构
    )
    templated_input_preview = templated_input_preview.replace("\n", "\\n")  # 换行转义成字面 \n，便于单行打印
    warmup_token_ids = warmup_result["token_ids"]  # 预热请求生成的 token 序列
    warmup_text = warmup_result["text"]  # 预热请求生成的文本
    print(
        "Warmup sample: "  # 打印预热样本的输入输出概览
        f"input={len(prompt_token_ids[0])}tok, "  # 输入 token 数
        f"templated_input_preview='{templated_input_preview}', "  # 模板化输入预览
        f"output={len(warmup_token_ids)}tok, "  # 输出 token 数
        f"preview='{warmup_text}'"  # 输出文本预览
    )

    t = time.time()  # 记录压测开始时刻
    bench_results = llm.generate(prompt_token_ids, sampling_params)  # 正式批量推理
    t = time.time() - t  # 总耗时（秒）

    total_output_budget = sum(sp.max_tokens for sp in sampling_params)  # 输出 token 预算上限 = 全部请求 max_tokens 之和
    output_lens = []  # 每条请求的实际输出长度
    for result in bench_results:  # 遍历每条结果
        token_ids = result["token_ids"]  # 取出生成的 token 序列
        output_lens.append(len(token_ids))  # 记录其长度
    total_output_tokens = sum(output_lens)  # 实际输出 token 总数

    print_len_stats("Input length", [len(x) for x in prompt_token_ids])  # 打印输入长度分布统计
    print_len_stats("Output length", output_lens)  # 打印输出长度分布统计
    throughput = total_output_tokens / t if t > 0 else 0.0  # 吞吐量 = 实际输出 token / 耗时（t>0 防除零）
    print(f"Bench requests: {len(prompt_token_ids)}")  # 打印压测请求数
    print(f"Output budget: {total_output_budget}tok, " f"Actual output: {total_output_tokens}tok")  # 预算 vs 实际（IGNORE_EOS=False 时实际会小于预算）
    print(f"Total: {total_output_tokens}tok, Time: {t:.2f}s, " f"Throughput: {throughput:.2f}tok/s")  # 打印最终吞吐结果


if __name__ == "__main__":  # 作为脚本直接运行时执行 main
    main()
