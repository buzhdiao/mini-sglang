from dataclasses import dataclass  # dataclass 装饰器，自动生成 __init__/__repr__ 等

import torch  # PyTorch：用于声明张量字段与创建设备上的缓冲区


@dataclass
class BaseCaptureData:
    """CUDA Graph 捕获期的静态元数据缓冲区基类。

    CUDA Graph 录制时会把 kernel 的输入地址固定下来，重放时不会重新分配内存，
    因此 decode 每次重放前必须把真实元数据"原地拷贝"进这些固定地址的缓冲区
    （shape 按 max_bs / max_seq_len 一次性分配到最大值）。
    各注意力后端派生自己的 CaptureData（如 FACaptureData），共享这套字段。
    """

    seq_lens: torch.Tensor  # 每个请求当前的 KV 序列长度（int32，形状 [max_bs]）
    positions: torch.Tensor  # 每个请求当前 token 的位置 id（int32，形状 [max_bs]），用于 RoPE 等
    cu_seqlens_k: torch.Tensor  # KV 侧累积序列长度前缀和（形状 [max_bs+1]），变长批次的"分界线"
    cu_seqlens_q: torch.Tensor  # Q 侧累积序列长度前缀和（形状 [max_bs+1]）；decode 时恒为 [0,1,2,...]
    page_table: torch.Tensor  # 页表：每行是一个请求的 KV 物理页/位置索引（形状 [max_bs, max_seq_len]）

    @classmethod  # 类方法工厂：按最大尺寸一次性创建全套静态缓冲区
    def create(cls, max_bs: int, max_seq_len: int, device: torch.device, **kwargs):
        return cls(  # 构造捕获期缓冲区集合；**kwargs 允许子类追加自定义字段
            seq_lens=torch.ones((max_bs,), dtype=torch.int32, device=device),  # 全 1 初始化：decode 阶段每请求恰好新增 1 个 token，天然适配默认值
            positions=torch.zeros((max_bs,), dtype=torch.int32, device=device),  # 位置 id 从 0 开始
            cu_seqlens_k=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),  # [0,1,...,max_bs]：decode 时每请求 KV 长度恰好差 1，前缀和天然成立，重放时只需部分覆盖
            cu_seqlens_q=torch.arange(0, max_bs + 1, dtype=torch.int32, device=device),  # decode 每请求 Q 长度为 1，前缀和同为 [0,1,...]，录制后无需更新（即 no-op）
            page_table=torch.zeros((max_bs, max_seq_len), dtype=torch.int32, device=device),  # 页表缓冲区，行数=最大 batch，列数=最大序列长度（以 token/page 为单位，由调用方约定）
            **kwargs,  # 透传子类额外字段
        )
