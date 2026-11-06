# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# gc 用于销毁图后强制回收对象
import gc
# dataclass 声明捕获缓冲类
from dataclasses import dataclass
# 类型注解工具
from typing import TYPE_CHECKING, Dict, List

# torch 提供 CUDA Graph 与张量
import torch
# 核心数据结构：批次与请求
from minisgl.core import Batch, Req, get_global_ctx
# TP 信息（判断主 rank 以决定是否打印进度条）
from minisgl.distributed import get_tp_info
# 日志工具
from minisgl.utils import init_logger
# tqdm 捕获过程进度条
from tqdm import tqdm

# 仅类型检查时导入
if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend
    from minisgl.models import BaseLLMModel

# 创建模块级 logger
logger = init_logger(__name__)


# CUDA Graph 捕获期的静态输入/输出缓冲：图回放时张量地址必须固定
@dataclass
class GraphCaptureBuffer:
    # 静态输入 token（int32）
    input_ids: torch.Tensor
    # 静态 KV 写入槽位
    out_loc: torch.Tensor
    # 静态位置 id
    positions: torch.Tensor
    # 静态 logits 输出缓冲（bs x vocab）
    logits: torch.Tensor

    # 类工厂：按最大批量分配一套静态缓冲
    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        # 全部缓冲一次性分配在目标设备上
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
        )

    # 捕获前：让批次直接"借用"静态缓冲的切片作为输入（地址固定进图）
    def set_batch(self, batch: Batch) -> None:
        # 只取填充批量对应的前缀
        _slice = slice(batch.padded_size)
        # 批次输入指向静态缓冲切片
        batch.input_ids = self.input_ids[_slice]
        # 槽位同样指向静态缓冲
        batch.out_loc = self.out_loc[_slice]
        # 位置同样指向静态缓冲
        batch.positions = self.positions[_slice]

    # 回放前：把真实批量的数据拷入静态缓冲（地址不变，图可复用）
    def copy_from(self, batch: Batch) -> None:
        # 填充批量的前缀范围
        _slice = slice(batch.padded_size)
        # 拷贝输入 token
        self.input_ids[_slice] = batch.input_ids
        # 拷贝写入槽位
        self.out_loc[_slice] = batch.out_loc
        # 拷贝位置 id
        self.positions[_slice] = batch.positions


# 决定要捕获 CUDA Graph 的批量列表：显式指定 > 按显存自动推断
def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    # 用户显式给定了批量列表则直接采用
    if cuda_graph_bs is not None:
        # 原样返回
        return cuda_graph_bs

    # 空闲显存换算成 GB
    free_memory_gb = free_memory / (1 << 30)
    # 未指定最大批量时按显存档位选
    if cuda_graph_max_bs is None:
        # 大显存卡（如 H200 80G+）可以捕获更大的批量
        if free_memory_gb > 80:  # H200
            # 最大 256
            cuda_graph_max_bs = 256
        else:
            # 其余按 160
            cuda_graph_max_bs = 160

    # 最大批量 <1 视为禁用图
    if cuda_graph_max_bs < 1:
        # 返回空列表
        return []

    # 小批量细粒度（1,2,4）+ 8 以上每 8 一档，覆盖 decode 常见并发
    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))


# 字节数格式化为 GiB 字符串
def mem_GB(size: int) -> str:
    # 除以 2^30 保留两位小数
    return f"{size / (1024**3):.2f} GiB"


# 查询设备当前空闲显存（字节数）
def get_free_memory(device: torch.device) -> int:
    # mem_get_info 返回 (free, total)，取 free
    return torch.cuda.mem_get_info(device)[0]


# CUDA Graph 执行器：负责捕获各 bs 的 decode 图与回放
class GraphRunner:
    # 保存依赖并立即开始捕获
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
    ) -> None:
        # 确定要捕获的批量列表
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        # 保存注意力后端（捕获/回放时需要配合准备元数据）
        self.attn_backend = attn_backend
        # 可用图的最大批量（空列表则为 0 表示禁用）
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        # 升序保存批量列表
        self.graph_bs_list = sorted(cuda_graph_bs)
        # 保存 dummy 请求（填充批量用）
        self.dummy_req = dummy_req
        # 保存执行流
        self.stream = stream
        # 保存设备
        self.device = device
        # 立即开始捕获全部图
        self._capture_graphs(max_seq_len, vocab_size, model)

    # 逐 bs 捕获 decode 的 CUDA Graph
    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # bs -> 图 的映射
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        # 批量列表为空：图被禁用
        if self.max_graph_bs == 0:
            # 打印提示并返回
            return logger.info_rank0("CUDA graph is disabled.")

        # 通知注意力后端初始化捕获缓冲（静态页表等）
        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        # 等设备空闲
        torch.cuda.synchronize(self.device)
        # 归还缓存块，测准显存
        torch.cuda.empty_cache()
        # 清零峰值统计
        torch.cuda.reset_peak_memory_stats(self.device)

        # 打印开始捕获信息（仅主 rank）
        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        # 记录捕获前空闲显存
        free_memory = get_free_memory(self.device)
        # 打印（仅主 rank）
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        # 按最大批量分配静态输入/输出缓冲
        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)

        # 进度条：从大到小逐个捕获（非主 rank 关闭显示）
        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        # 图内存池：第一个图创建后，后续图复用同一池减少显存碎片
        pool = None
        # 逐批量捕获
        for bs in pbar:
            # 每次捕获前刷新空闲显存
            free_memory = get_free_memory(self.device)
            # 更新进度条描述：当前 bs 与可用显存
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            # 立即刷新显示
            pbar.refresh()
            # 创建图对象
            graph = torch.cuda.CUDAGraph()
            # 构造全 dummy 的 decode 批（bs 个 dummy 请求）
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            # 填充请求就是请求本身（无额外 padding）
            batch.padded_reqs = batch.reqs
            # 让注意力后端把静态元数据挂上批次
            self.attn_backend.prepare_for_capture(batch)
            # 批次的输入直接指向静态缓冲切片
            self.buffer.set_batch(batch)
            # 登记当前批次到全局上下文
            with get_global_ctx().forward_batch(batch):
                # 图外先跑一次：完成 shape 推导并写入 logits 缓冲
                self.buffer.logits[:bs] = model.forward()
                # 进入图捕获模式（在专用流上）
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    # 再次执行 model.forward：此时 kernel 启动被记录进图
                    self.buffer.logits[:bs] = model.forward()
            if pool is None:
                # 复用第一个图的内存池句柄，避免每个图各留一份中间缓冲（见行尾注释）
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            # 记录 bs -> 图
            self.graph_map[bs] = graph

        # 捕获完成后的空闲显存
        free_memory = get_free_memory(self.device)
        # 打印（仅主 rank）
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    # 判断该批次能否走 CUDA Graph：decode 批且 bs <= 最大图批量
    def can_use_cuda_graph(self, batch: Batch) -> bool:
        # decode 且批量不超过已捕获的最大值
        return batch.is_decode and batch.size <= self.max_graph_bs

    # 回放图：拷数据 -> 准备元数据 -> replay -> 返回真实部分的 logits
    def replay(self, batch: Batch) -> torch.Tensor:
        # 必须满足可用图条件
        assert self.can_use_cuda_graph(batch)
        # 把真实批量的动态输入拷进静态缓冲
        self.buffer.copy_from(batch)
        # 取该填充批量对应的图
        g = self.graph_map[batch.padded_size]
        # 注意力后端把真实元数据拷入捕获期静态缓冲
        self.attn_backend.prepare_for_replay(batch)
        # 一次 replay 重放整段 decode 计算
        g.replay()
        # 返回真实（未填充）部分的 logits
        return self.buffer.logits[: batch.size]

    # 把批量填充到最近的可用图批量（decode 调度阶段调用）
    def pad_batch(self, batch: Batch) -> None:
        # choose the first available batch size
        # 选第一个 >= 实际批量的捕获批量；不能用图时按原样
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        # 用 dummy 请求补齐到填充批量
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    # 销毁全部图：必须在释放 NCCL 资源之前调用，否则可能挂死
    def destroy_cuda_graphs(self) -> None:
        # 删除 bs -> 图 的映射
        del self.graph_map
        # 立即触发垃圾回收释放图资源
        gc.collect()
