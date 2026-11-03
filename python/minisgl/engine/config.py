from __future__ import annotations  # 启用延迟注解求值，允许在注解中使用尚未定义的类型（如 List[int] | None）

from dataclasses import dataclass  # dataclass 装饰器：自动生成 __init__/__repr__ 等方法
from functools import cached_property  # cached_property：首次访问后缓存结果的属性，避免重复计算
from typing import TYPE_CHECKING, List  # TYPE_CHECKING：仅类型检查时导入，避免运行时循环依赖；List：列表泛型

import torch  # PyTorch 主库，这里主要用其 dtype 类型（如 torch.bfloat16）
from minisgl.distributed import DistributedInfo  # 分布式（TP）信息：rank、world_size 等
from minisgl.utils import cached_load_hf_config  # 带磁盘缓存的 HuggingFace config.json 加载工具，避免重复读盘

if TYPE_CHECKING:  # 仅在静态类型检查时生效，运行时不执行，规避循环导入
    from minisgl.models import ModelConfig  # 仅作返回类型注解使用


@dataclass(frozen=True)  # frozen=True：实例不可变，配置创建后不允许被修改（保证配置安全）
class EngineConfig:
    """引擎配置：汇聚启动推理引擎所需的全部参数，并提供若干派生属性的只读视图"""

    model_path: str  # HuggingFace 模型目录路径（含 config.json 与权重文件）
    tp_info: DistributedInfo  # 张量并行信息：本进程 rank 与 TP 世界大小
    dtype: torch.dtype  # 计算与存储精度（如 torch.bfloat16），决定权重与 KV Cache 的元素类型
    max_running_req: int = 256  # 引擎可同时容纳的最大运行请求数（决定页表行数）
    attention_backend: str = "auto"  # 注意力后端选择；"auto" 表示按 GPU 架构自动挑选
    moe_backend: str = "auto"  # MoE（专家混合）后端选择；"auto" 表示自动挑选
    cuda_graph_bs: List[int] | None = None  # 显式指定要捕获 CUDA Graph 的 batch size 列表；None 为自动
    cuda_graph_max_bs: int | None = None  # 自动模式下 CUDA Graph 支持的最大 batch size；None 按显存推断
    page_size: int = 1  # KV Cache 分页大小（每页 token 数）；1 即逐 token 分配，无页内浪费
    memory_ratio: float = 0.9  # 分配给 KV Cache 的显存比例（占总空闲显存）
    distributed_timeout: float = 60.0  # 进程组通信超时秒数，防止某 rank 挂死时整体无限等待
    use_dummy_weight: bool = False  # True 时用随机权重替代真实权重（用于性能测试/调试，跳过加载）
    use_pynccl: bool = True  # 是否启用 pynccl 自定义通信路径（在 CPU 组上再挂 NCCL）
    max_seq_len_override: int | None = None  # 手动覆盖最大序列长度；None 则取模型配置值
    num_page_override: int | None = None  # if not None, will override the number of pages  # 手动指定 KV Cache 页数；None 则按显存自动计算

    @cached_property  # 首次访问时加载并缓存 HF 配置，后续访问直接命中缓存
    def hf_config(self):
        return cached_load_hf_config(self.model_path)  # 从模型路径读取（或用缓存）HuggingFace 配置对象

    @cached_property  # 首次访问时构建并缓存模型配置对象
    def model_config(self) -> ModelConfig:
        from minisgl.models import ModelConfig  # 函数内延迟导入：打破模块级循环依赖

        return ModelConfig.from_hf(self.hf_config)  # 从 HF 配置提炼出 minisgl 自己的模型配置

    @property  # 只读派生属性，每次访问重新求值
    def max_seq_len(self) -> int:
        if self.max_seq_len_override is not None:  # 用户显式指定了最大序列长度
            return self.max_seq_len_override  # 直接使用覆盖值
        return self.model_config.rotary_config.max_position  # 否则取 RoPE 配置支持的最大位置数

    @property  # 一次前向允许处理的最大 token 数
    def max_forward_len(self) -> int:
        return self.max_seq_len  # 取最大序列长度（用于估算通信缓冲区大小等）

    @property  # 分布式初始化用的 TCP 地址（单机多卡场景，各 rank 经此 rendezvous）
    def distributed_addr(self) -> str:
        return "tcp://127.0.0.1:2333"  # 固定本机 2333 端口作为进程组初始化入口
