# shutil 用于把下载的远程流拷贝到本地文件
import shutil
# time 统计整体压测耗时
import time
# urllib.request 通过 HTTP 下载数据集分片
import urllib.request
# Path 做路径拼接与存在性判断
from pathlib import Path
# seed 固定随机种子
from random import seed

# pyarrow.parquet 分批读取 WildChat 数据集的 parquet 分片
import pyarrow.parquet as pq
# 采样参数定义
from minisgl.core import SamplingParams
# 离线推理入口
from minisgl.llm import LLM
# AutoTokenizer 加载与模型配套的分词器
from transformers import AutoTokenizer

# 只保留英语/中文的真实用户 prompt，过滤小语种噪声
LANGS = {"English", "Chinese"}

# 数据分片下载到脚本所在目录
BASE_DIR = Path(__file__).resolve().parent
# 只取 WildChat-4.8M 的第一个分片（约 1/86 的数据，足够压测）
WILDCHAT_FIRST_SHARD = "train-00000-of-00086.parquet"
# 数据集文件的基础 URL
WILDCHAT_BASE_URL = "https://huggingface.co/datasets/allenai/WildChat-4.8M/resolve/main/data/"


# 若目标文件不存在则从 URL 下载到本地（幂等）
def download_if_missing(url: str, path: Path) -> None:
    # 已下载过则直接跳过，支持重复运行
    if path.exists():
        return
    # 提示下载进度
    print(f"Downloading {url} -> {path}")
    # 确保目标目录存在
    path.parent.mkdir(parents=True, exist_ok=True)
    # 打开远程响应流与本地输出文件
    with urllib.request.urlopen(url, timeout=300) as resp, path.open("wb") as out:
        # 边读边写，避免一次性把大文件读进内存
        shutil.copyfileobj(resp, out)


# 生成器：逐批读取 parquet，清洗出真实用户 prompt 并编码成 token id
def iter_filtered_prompt_ids(tokenizer):
    # 本地分片路径
    shard_path = BASE_DIR / WILDCHAT_FIRST_SHARD
    # 缺文件时先下载
    download_if_missing(WILDCHAT_BASE_URL + WILDCHAT_FIRST_SHARD, shard_path)

    # 打开 parquet 文件（支持分批迭代，不整表载入内存）
    parquet = pq.ParquetFile(shard_path)
    # 每批 256 行，只读 conversation 列（每行是一条多轮对话）
    for batch in parquet.iter_batches(batch_size=256, columns=["conversation"]):
        # 收集本批通过过滤的 prompt 文本
        prompts = []
        # 遍历每条会话（若干轮 message）
        for conv in batch.to_pydict()["conversation"]:
            # 空会话直接跳过
            if not conv:
                continue

            # 记录第一条用户消息，作为压测输入
            first_user = None
            # 按顺序扫描各轮消息
            for turn in conv:
                # role 为 user 的轮次即用户输入
                if turn.get("role") == "user":
                    # 命中后立即记录
                    first_user = turn
                    # 只取第一条，模拟"首问"场景
                    break
            # 没有用户消息的会话跳过
            if first_user is None:
                continue

            # 取消息正文并去首尾空白（content 字段可能为 None）
            text = (first_user.get("content") or "").strip()
            # 空文本跳过
            if not text:
                continue
            # 非英语/中文跳过，保证输入长度分布合理
            if first_user.get("language") not in LANGS:
                continue
            # 被脱敏（redacted）或有毒（toxic）标记的消息跳过
            if bool(first_user.get("redacted")) or bool(first_user.get("toxic")):
                continue
            # 通过全部过滤条件，收入本批结果
            prompts.append(text)

        # 本批没有可用 prompt 则继续下一批
        if not prompts:
            continue

        # 对每条可用 prompt 做 chat template 编码
        for prompt in prompts:
            # 按模型对话模板把单条用户消息转成 token 序列
            ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            # 逐条产出 token id，供主流程按需取用
            yield ids


# 打印一组长度的分位数统计
def print_len_stats(name: str, lengths: list[int]) -> None:
    # 空列表直接说明无数据
    if not lengths:
        print(f"{name}: no data")
        return
    # 升序排序便于按下标取分位数
    arr = sorted(lengths)
    # 样本数
    n = len(arr)
    # 输出 count/min/p50/p90/p99/max，观察真实负载的长度分布（p90/p99 反映长尾）
    print(
        f"{name}: count={n}, min={arr[0]}, p50={arr[int(0.50*n)]}, "
        f"p90={arr[int(0.90*n)]}, p99={arr[int(0.99*n)]}, max={arr[-1]}"
    )


# 用 WildChat 真实 prompt 做离线批量压测
def main() -> None:
    # 固定随机种子，可复现
    seed(0)
    # 被测模型
    MODEL = "Qwen/Qwen3-0.6B"
    # 压测请求条数
    NUM_SEQS = 256 * 1
    # 每条请求的最大输出长度
    MAX_OUTPUT_LEN = 4096
    # 不忽略 EOS：保留真实停止行为，输出长度才贴近真实分布
    IGNORE_EOS = False

    # 加载分词器，用于 chat template 编码与结果解码
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # 收集压测输入
    prompt_token_ids = []
    # 从数据集逐条取 prompt 的 token id
    for ids in iter_filtered_prompt_ids(tokenizer):
        # 攒够请求数为止
        prompt_token_ids.append(ids)
        # 达到请求数上限即停止读取
        if len(prompt_token_ids) >= NUM_SEQS:
            break

    # 实际收集到的请求数（可能少于 NUM_SEQS）
    num_reqs = len(prompt_token_ids)
    # 为每条请求生成相同的采样配置
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=IGNORE_EOS, max_tokens=MAX_OUTPUT_LEN)
        for _ in range(num_reqs)
    ]

    # 创建引擎（用默认超参加载模型）
    llm = LLM(MODEL)

    # warm up
    # 预热：先跑一条请求，触发内核初始化/JIT 编译等一次性开销，避免计入正式压测
    warmup_result = llm.generate(
        [prompt_token_ids[0]],
        sampling_params[0],
    )[0]
    # 把输入 token id 解码回字符串，便于人工检查 chat 模板
    templated_input_preview = tokenizer.decode(
        prompt_token_ids[0],
        skip_special_tokens=False,
    )
    # 换行转义成字面量，方便单行打印
    templated_input_preview = templated_input_preview.replace("\n", "\\n")
    # 预热请求的输出 token id
    warmup_token_ids = warmup_result["token_ids"]
    # 预热请求的输出文本
    warmup_text = warmup_result["text"]
    # 打印预热样本的输入/输出概况，确认数据与模板正常
    print(
        "Warmup sample: "
        f"input={len(prompt_token_ids[0])}tok, "
        f"templated_input_preview='{templated_input_preview}', "
        f"output={len(warmup_token_ids)}tok, "
        f"preview='{warmup_text}'"
    )

    # 记录压测开始时间
    t = time.time()
    # 提交全部请求，由引擎 continuous batching 调度执行
    bench_results = llm.generate(prompt_token_ids, sampling_params)
    # 计算总耗时
    t = time.time() - t

    # 理论最大输出预算（每条按 MAX_OUTPUT_LEN 计）
    total_output_budget = sum(sp.max_tokens for sp in sampling_params)
    # 收集每条请求实际输出长度
    output_lens = []
    # 遍历各请求结果
    for result in bench_results:
        # 取输出 token id
        token_ids = result["token_ids"]
        # 记录该请求的输出长度
        output_lens.append(len(token_ids))
    # 实际输出 token 总数
    total_output_tokens = sum(output_lens)

    # 打印输入长度分布
    print_len_stats("Input length", [len(x) for x in prompt_token_ids])
    # 打印输出长度分布
    print_len_stats("Output length", output_lens)
    # 端到端吞吐（按实际输出 token 计），耗时为 0 时给 0 防除零
    throughput = total_output_tokens / t if t > 0 else 0.0
    # 打印压测请求数
    print(f"Bench requests: {len(prompt_token_ids)}")
    # 对比输出预算与实际输出量
    print(f"Output budget: {total_output_budget}tok, " f"Actual output: {total_output_tokens}tok")
    # 打印最终吞吐结果
    print(f"Total: {total_output_tokens}tok, Time: {t:.2f}s, " f"Throughput: {throughput:.2f}tok/s")


# 直接运行脚本时执行
if __name__ == "__main__":
    # 入口
    main()
