# dataclass 简化捕获数据类的声明
from dataclasses import dataclass

# torch 提供 GPU 张量与设备管理
import torch


# CUDA Graph 捕获期间使用的静态缓冲区集合：
# 捕获时张量地址必须固定，回放时把真实数据拷入这些缓冲即可
@dataclass
class BaseCaptureData:
    # 每个请求的 KV 序列长度（decode 阶段每步更新）
    seq_lens: torch.Tensor
    # 每个请求最后一个 token 的位置 id（RoPE 等用）
    positions: torch.Tensor
    # KV 的 cumulative 序列长度（ragged 布局：[0, len1, len1+len2, ...]）
    cu_seqlens_k: torch.Tensor
    # Q 的 cumulative 序列长度（decode 时恒为 [0,1,2,...]）
    cu_seqlens_q: torch.Tensor
    # 每请求的页表/位置索引（行对应请求，列对应 token 槽位）
    page_table: torch.Tensor

    # 类工厂：按最大批量与最大序列长度一次性分配好捕获期静态缓冲。
    # seq_lens 全 1（每请求至少 1 个 KV token）；positions 全 0；
    # cu_seqlens 初始为 [0..max_bs]；页表缓冲为 max_bs 行 x max_seq_len 列
    @classmethod
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        # 构造各缓冲区，全部为 int32 以减少 H2D 拷贝带宽
        return cls(
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),
            **kwargs,
        )
