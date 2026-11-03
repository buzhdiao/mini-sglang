from __future__ import annotations  # 启用延迟注解求值，允许前向引用类型

import gc  # 垃圾回收模块：销毁 CUDA Graph 时手动触发回收，确保显存及时释放
from dataclasses import dataclass  # dataclass 装饰器：自动生成样板方法
from typing import TYPE_CHECKING, Dict, List  # 类型注解工具；TYPE_CHECKING 用于仅类型检查时导入

import torch  # PyTorch：CUDA Graph、张量、设备管理
from minisgl.core import Batch, Req, get_global_ctx  # 核心结构：批次、请求、全局上下文读取器
from minisgl.distributed import get_tp_info  # 获取 TP 信息（判断是否为主 rank 以控制进度条）
from minisgl.utils import init_logger  # 日志工具
from tqdm import tqdm  # 终端进度条：展示图捕获进度

if TYPE_CHECKING:  # 仅类型检查时导入，避免运行时循环依赖
    from minisgl.attention import BaseAttnBackend  # 注意力后端基类（仅作注解）
    from minisgl.models import BaseLLMModel  # 模型基类（仅作注解）

logger = init_logger(__name__)  # 初始化模块日志器


@dataclass  # 数据类装饰器：自动生成 __init__/__repr__
class GraphCaptureBuffer:
    """CUDA Graph 捕获/重放共用的固定地址输入输出缓冲区：图内张量地址固定，重放时只需原地拷贝新数据"""

    input_ids: torch.Tensor  # 输入 token id 缓冲（形状 [max_bs]）
    out_loc: torch.Tensor  # 每个请求新 token 的 KV 写入物理位置（形状 [max_bs]）
    positions: torch.Tensor  # 每个请求当前 token 的序列位置（形状 [max_bs]）
    logits: torch.Tensor  # 输出 logits 缓冲（形状 [max_bs, vocab_size]）

    @classmethod  # 类方法作为命名构造器
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(  # 按最大 batch size 一次性分配全部缓冲
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),  # 零填充的输入 id
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),  # 零填充的写入位置
            positions=torch.zeros(bs, dtype=torch.int32, device=device),  # 零填充的位置索引
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),  # 空的 logits（内容每次前向覆盖）
        )

    def set_batch(self, batch: Batch) -> None:
        """捕获阶段：把缓冲区的切片视图赋给批次，使图内引用的就是这些固定地址"""
        _slice = slice(batch.padded_size)  # 取前 padded_size 行（填充后的大小）
        batch.input_ids = self.input_ids[_slice]  # 批次输入 id 直接引用缓冲切片
        batch.out_loc = self.out_loc[_slice]  # 写入位置同理
        batch.positions = self.positions[_slice]  # 位置索引同理

    def copy_from(self, batch: Batch) -> None:
        """重放阶段：把真实批次数据原地拷入缓冲区（地址不变，图内kernel即可读到新数据）"""
        _slice = slice(batch.padded_size)  # 只覆盖有效行
        self.input_ids[_slice] = batch.input_ids  # 拷入输入 id
        self.out_loc[_slice] = batch.out_loc  # 拷入写入位置
        self.positions[_slice] = batch.positions  # 拷入位置索引


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,  # 用户显式指定的 bs 列表
    cuda_graph_max_bs: int | None,  # 自动模式的最大 bs
    free_memory: int,  # 可用显存字节数（用于推断默认上限）
) -> List[int]:
    """确定要捕获的 batch size 列表：显式指定优先，否则按显存自动生成"""
    if cuda_graph_bs is not None:  # 用户给了显式列表
        return cuda_graph_bs  # 原样采用

    free_memory_gb = free_memory / (1 << 30)  # 字节转 GiB
    if cuda_graph_max_bs is None:  # 未指定最大 bs：按显存档次推断
        if free_memory_gb > 80:  # H200  # 大显存卡（>80GiB）
            cuda_graph_max_bs = 256  # 支持到 bs=256
        else:
            cuda_graph_max_bs = 160  # 常规卡支持到 bs=160

    if cuda_graph_max_bs < 1:  # 非法/禁用（<1）时
        return []  # 返回空列表表示完全禁用 CUDA Graph

    return [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))  # 小 bs 密集（1/2/4），此后步长 8 递增到上限


def mem_GB(size: int) -> str:
    return f"{size / (1024**3):.2f} GiB"  # 字节数格式化为保留两位小数的 GiB 字符串


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]  # 返回该设备的空闲显存字节数


class GraphRunner:
    """CUDA Graph 运行器：按离散 bs 列表捕获 decode 前向图，运行时选不小于实际 bs 的最小图重放"""

    def __init__(
        self,
        stream: torch.cuda.Stream,  # 捕获/重放使用的 CUDA 流
        device: torch.device,  # 目标设备
        model: BaseLLMModel,  # 待捕获的模型
        attn_backend: BaseAttnBackend,  # 注意力后端（捕获/重放需配合准备元数据）
        cuda_graph_bs: List[int] | None,  # 显式 bs 列表
        cuda_graph_max_bs: int | None,  # 自动模式最大 bs
        free_memory: int,  # 初始空闲显存（推断上限用）
        max_seq_len: int,  # 对齐后的最大序列长度（后端预留捕获空间）
        vocab_size: int,  # 词表大小（logits 缓冲宽度）
        dummy_req: Req,  # 哑请求：填充与捕获的占位
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(  # 先把 bs 列表确定下来
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend  # 保存注意力后端引用
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0  # 已捕获的最大 bs；空列表记 0 表示禁用
        self.graph_bs_list = sorted(cuda_graph_bs)  # 排序后的 bs 列表（便于按序捕获与查找）
        self.dummy_req = dummy_req  # 保存哑请求
        self.stream = stream  # 保存流
        self.device = device  # 保存设备
        self._capture_graphs(max_seq_len, vocab_size, model)  # 构造即捕获（启动开销前置到初始化）

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        """逐个 bs 捕获 decode 前向的 CUDA Graph，全部共享一个显存池"""
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}  # bs -> 已捕获图的映射
        if self.max_graph_bs == 0:  # bs 列表为空：用户禁用了 CUDA Graph
            return logger.info_rank0("CUDA graph is disabled.")  # 打印提示并直接返回

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)  # 通知后端进入捕获模式并预留元数据空间

        torch.cuda.synchronize(self.device)  # 同步确保无残留 kernel
        torch.cuda.empty_cache()  # 清缓存，给图捕获腾出连续显存
        torch.cuda.reset_peak_memory_stats(self.device)  # 重置峰值统计

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")  # 打印捕获计划
        free_memory = get_free_memory(self.device)  # 查询捕获前空闲显存
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")  # 打印捕获前余量

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)  # 按最大 bs 分配共享输入输出缓冲

        pbar = tqdm(  # 进度条按 bs 从大到小遍历
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks  # 非主 rank 关闭进度条避免刷屏
        )
        pool = None  # CUDA Graph 显存池：首个图创建后其余图复用，省显存
        for bs in pbar:
            free_memory = get_free_memory(self.device)  # 每轮查询剩余显存
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"  # 动态更新进度条描述
            pbar.refresh()  # 立即刷新显示
            graph = torch.cuda.CUDAGraph()  # 新建一个空图对象
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")  # 用 bs 个哑请求构造捕获用解码批
            batch.padded_reqs = batch.reqs  # 捕获时无填充需求：padded 即 reqs 本身
            self.attn_backend.prepare_for_capture(batch)  # 后端为该批准备捕获期的注意力元数据
            self.buffer.set_batch(batch)  # 把批次输入指向共享缓冲的固定地址
            with get_global_ctx().forward_batch(batch):  # 前向上下文：登记当前批，模型内部可取页表等
                self.buffer.logits[:bs] = model.forward()  # 温跑一次前向：让 lazy 初始化（如编译/缓存）发生在捕获前
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):  # 进入图捕获模式（在此流上记录 kernel）
                    self.buffer.logits[:bs] = model.forward()  # 正式捕获的前向：kernel 被录进 graph
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory  # 首图之后所有图共享同一显存池
            self.graph_map[bs] = graph  # 登记 bs -> 图

        free_memory = get_free_memory(self.device)  # 捕获完成后查询显存
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")  # 打印捕获后余量

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode and batch.size <= self.max_graph_bs  # 仅解码批且规模不超过已捕获最大 bs 时可用图

    def replay(self, batch: Batch) -> torch.Tensor:
        """用预捕获的图执行一次前向：拷数据 -> 准备元数据 -> 重放 -> 返回 logits"""
        assert self.can_use_cuda_graph(batch)  # 调用方必须保证可用（内部断言兜底）
        self.buffer.copy_from(batch)  # 把真实输入原地拷入共享缓冲（地址与图内一致）
        g = self.graph_map[batch.padded_size]  # 取填充后大小对应的图
        self.attn_backend.prepare_for_replay(batch)  # 后端更新重放期的注意力元数据（页表指针等）
        g.replay()  # 一次性重放整图 kernel
        return self.buffer.logits[: batch.size]  # 返回有效行（去掉填充）的 logits

    def pad_batch(self, batch: Batch) -> None:
        """把批次填充到已捕获的某个图大小（用哑请求补位）"""
        padded_size = (  # choose the first available batch size  # 选第一个 ≥ 实际大小的已捕获 bs
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size  # 不能用图时无需填充
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)  # 尾部补哑请求到目标大小

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    # 必须在释放 NCCL 资源之前销毁图，否则程序可能挂死
    def destroy_cuda_graphs(self) -> None:
        del self.graph_map  # 删除图映射，释放对 CUDAGraph 的引用
        gc.collect()  # 手动触发 GC，确保图对象立即析构、显存即时归还


