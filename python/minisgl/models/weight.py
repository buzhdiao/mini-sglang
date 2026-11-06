# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# glob 用于枚举权重分片文件
import glob
# re 用于解析专家权重键名
import re
from typing import Dict, Iterator, Tuple

# safetensors 高效安全地读取权重文件
import safetensors
import torch
# TP 并行信息：按 rank 切分权重
from minisgl.distributed import get_tp_info
# HF 配置缓存读取、向上取整、权重下载
from minisgl.utils import cached_load_hf_config, div_ceil, download_hf_weight
# tqdm 进度条
from tqdm import tqdm

# 按 0 维（输出维）切分的投影：列并行
_SPLIT_DIM_0 = [".q_proj", ".k_proj", ".v_proj", ".gate_proj", ".up_proj"]
# 按 1 维（输入维）切分的投影：行并行
_SPLIT_DIM_1 = [".o_proj", ".down_proj"]

# Merge groups: individual projections -> fused projection
# 合并组：检查点里的独立投影 -> 运行时融合投影（QKV 合一、gate/up 合一）
_MERGE_GROUPS = {
    ".q_proj": (".qkv_proj", ("q", "k", "v")),
    ".k_proj": (".qkv_proj", ("q", "k", "v")),
    ".v_proj": (".qkv_proj", ("q", "k", "v")),
    ".gate_proj": (".gate_up_proj", ("gate", "up")),
    ".up_proj": (".gate_up_proj", ("gate", "up")),
}
# 各投影在融合权重里的槽位名
_SLOT_NAMES = {
    ".q_proj": "q",
    ".k_proj": "k",
    ".v_proj": "v",
    ".gate_proj": "gate",
    ".up_proj": "up",
}
# 匹配专家权重键名的正则：<前缀>.experts.<专家编号>.<名称>
_EXPERT_PATTERN = re.compile(r"^(?P<prefix>.+\.experts)\.(?P<idx>\d+)\.(?P<name>.+)$")


def _shard_tensor(key: str, value: torch.Tensor, r: int, n: int, num_kv_heads: int):
    """Extract rank r's shard from a single tensor. Returns a contiguous copy."""
    # 从整份权重里取出 rank r 的分片（返回连续副本，避免跨进程引用大缓冲）
    if any(key.count(sub) for sub in _SPLIT_DIM_0):
        # 判断是否为 KV 投影
        is_kv_proj = any(key.count(sub) for sub in (".k_proj", ".v_proj"))
        if is_kv_proj and num_kv_heads is not None and num_kv_heads < n:
            # GQA 且 KV 头数少于 rank 数：按"KV 头"粒度复制给多个 rank，
            # 让每 rank 拿整数个 KV 头（相邻 rank 可能共享同一头），而非任意切行
            head_dim = value.shape[0] // num_kv_heads
            # rank r 对应的第几个 KV 头
            head_idx = r * num_kv_heads // n
            return value[head_idx * head_dim : (head_idx + 1) * head_dim].clone()
        # 一般列并行：按 0 维均切成 n 份取第 r 份
        return value.chunk(n, dim=0)[r].clone()
    elif any(key.count(sub) for sub in _SPLIT_DIM_1):
        # 行并行：按 1 维（输入维）均切
        return value.chunk(n, dim=1)[r].clone()
    elif key.count("lm_head") or key.count("embed_tokens"):
        # 词表并行：按词表行切分（向上取整，末位 rank 可能不满）
        num_embeddings = value.shape[0]
        num_embeddings_per_partition = div_ceil(num_embeddings, n)
        vocab_start_idx = r * num_embeddings_per_partition
        vocab_end_idx = min((r + 1) * num_embeddings_per_partition, num_embeddings)
        return value[vocab_start_idx:vocab_end_idx, :].clone()
    else:
        # 其余权重（LayerNorm、路由门控等）不切分，各 rank 复制完整份
        return value


def _get_merge_info(key: str):
    """If key belongs to a merge group, return (merged_key, slot, all_slots). Else None."""
    # 判断该键是否属于某个融合组：是则返回 (融合后的键, 本槽位名, 全部槽位名)
    for suffix, (fused_suffix, slots) in _MERGE_GROUPS.items():
        if key.count(suffix):
            return key.replace(suffix, fused_suffix), _SLOT_NAMES[suffix], slots
    return None


def _get_expert_stack_info(key: str) -> tuple[str, int] | None:
    """Map an expert-scoped checkpoint key to the packed runtime key."""
    # 把"单个专家"的检查点键映射为运行时"专家堆叠"键 + 专家编号
    match = _EXPERT_PATTERN.match(key)
    if match is None:
        return None

    # 取出权重名（如 gate_proj / up_proj / down_proj）
    packed_name = match.group("name")
    if packed_name.endswith(".weight"):
        packed_name = packed_name.removesuffix(".weight")
    # 返回 (打包键前缀, 专家编号)
    return f"{match.group('prefix')}.{packed_name}", int(match.group("idx"))


def load_weight(model_path: str, device: torch.device) -> Iterator[Tuple[str, torch.Tensor]]:
    """Streaming weight loader. Yields (name, tensor) pairs already sharded, merged,
    and on device. Peak CPU memory: one full tensor + a small merge buffer."""
    # 流式权重加载器：逐张量产出"已切分/已合并/已在目标设备上"的 (名字, 张量)，
    # 峰值内存 = 单个完整张量 + 小的合并缓冲，避免整模型驻留 CPU 内存
    from .config import ModelConfig

    # 下载（或定位本地）HF 权重目录
    model_folder = download_hf_weight(model_path)
    # 读取模型配置（取 KV 头数等切分信息）
    config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    # 枚举所有 safetensors 分片
    files = glob.glob(f"{model_folder}/*.safetensors")
    # 优先跳过旧版全量 consolidated 文件（分片文件已覆盖全部权重）
    files = [f for f in files if not f.endswith("consolidated.safetensors")] or files
    tp_info = get_tp_info()

    # Buffer for merge groups: merged_key -> {slot: tensor}
    # 融合组缓冲：等 Q/K/V（或 gate/up）到齐后再拼接产出
    merge_buf: Dict[str, Dict[str, torch.Tensor]] = {}
    # 专家堆叠缓冲：等全部专家到齐后再 stack 产出
    expert_buf: Dict[str, Dict[int, torch.Tensor]] = {}
    # 仅主 rank 显示进度条（其余 rank 也在各自加载自己的分片）
    for file in tqdm(files, desc="Loading weights", disable=not tp_info.is_primary()):
        # 直接把张量读到目标设备，省一次 CPU->GPU 拷贝
        with safetensors.safe_open(file, framework="pt", device=str(device)) as f:
            for name in f.keys():
                # Strip multimodal wrapper prefix, skip vision/projector weights
                # 跳过多模态模型的视觉塔/投影器权重（只加载语言模型部分）
                if name.startswith(("vision_tower.", "multi_modal_projector.")):
                    continue
                # 读出原始完整张量
                raw = f.get_tensor(name)
                # 去掉多模态包装前缀，得到语言模型的标准键名
                name = name.removeprefix("language_model.")
                # 按 TP 规则切出本 rank 的分片
                tensor = _shard_tensor(name, raw, tp_info.rank, tp_info.size, config.num_kv_heads)
                # 释放原始张量引用，控制内存峰值
                del raw

                if (info := _get_merge_info(name)) is None:
                    # 不属于融合组：直接作为产出候选
                    out = (name, tensor)
                else:
                    # 属于融合组：先暂存对应槽位
                    merged_key, slot, all_slots = info
                    merge_buf.setdefault(merged_key, {})[slot] = tensor
                    if not all(s in merge_buf[merged_key] for s in all_slots):
                        # 组内还没到齐，继续等下一个张量
                        continue
                    # 到齐后按固定槽位顺序沿 0 维拼接（Q|K|V 或 gate|up）
                    parts = [merge_buf[merged_key][s] for s in all_slots]
                    del merge_buf[merged_key]
                    out = (merged_key, torch.cat(parts, dim=0))

                if config.is_moe and (expert_info := _get_expert_stack_info(out[0])) is not None:
                    # MoE 模型：把单专家权重按 (专家数, ...) 维度堆叠成运行时布局
                    packed_key, expert_idx = expert_info
                    slots = expert_buf.setdefault(packed_key, {})
                    slots[expert_idx] = out[1]
                    if len(slots) != config.num_experts:
                        # 专家还没收齐，继续等
                        continue
                    # 按专家编号顺序取出全部分片
                    experts = [slots[idx] for idx in range(config.num_experts)]
                    del expert_buf[packed_key]
                    # 产出堆叠后的专家张量（供融合 MoE 内核一次处理全部专家）
                    yield packed_key, torch.stack(experts, dim=0)
                else:  # Normal dense model
                    # 普通稠密模型：直接产出
                    yield out[0], out[1]

    assert not merge_buf, f"Incomplete merge groups in checkpoint: {list(merge_buf.keys())}"
    assert not expert_buf, f"Incomplete expert tensors in checkpoint: {list(expert_buf.keys())}"
    # 收尾校验：融合组与专家缓冲必须为空，否则检查点缺权重
