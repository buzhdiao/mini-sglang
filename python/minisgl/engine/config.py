# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# dataclass 声明配置类
from dataclasses import dataclass
# cached_property 缓存只需解析一次的派生配置
from functools import cached_property
# 类型注解工具
from typing import TYPE_CHECKING, List

# torch 提供数据类型对象
import torch
# TP 进程组信息
from minisgl.distributed import DistributedInfo
# 带缓存的 HF 配置加载（多进程共享解析结果）
from minisgl.utils import cached_load_hf_config

# 仅类型检查时导入模型配置
if TYPE_CHECKING:
    from minisgl.models import ModelConfig


# 引擎启动配置（frozen：初始化流程不允许随意改动）
@dataclass(frozen=True)
class EngineConfig:
    # 模型路径（本地目录或 HF hub 名）
    model_path: str
    # 本进程的 TP 信息（rank/size）
    tp_info: DistributedInfo
    # 计算与 KV cache 的数据类型
    dtype: torch.dtype
    # 同时运行的最大请求数（决定页表行数）
    max_running_req: int = 256
    # 注意力后端（auto 表示按 GPU 架构自动选择）
    attention_backend: str = "auto"
    # MoE 后端（auto 表示自动选择）
    moe_backend: str = "auto"
    # 显式指定要捕获 CUDA Graph 的批量列表
    cuda_graph_bs: List[int] | None = None
    # 只给最大批量，其余自动选取
    cuda_graph_max_bs: int | None = None
    # KV cache 分页大小（token 数/页）
    page_size: int = 1
    # 允许 KV cache 占用的显存比例
    memory_ratio: float = 0.9
    # 分布式初始化/通信超时（秒）
    distributed_timeout: float = 60.0
    # 使用随机权重代替真实权重（调试/性能测试用）
    use_dummy_weight: bool = False
    # 是否启用自研 PyNCCL 通信通道
    use_pynccl: bool = True
    # 覆盖模型的最大序列长度
    max_seq_len_override: int | None = None
    # if not None, will override the number of pages
    # 显式指定 KV cache 页数（调试/受控实验用）
    num_page_override: int | None = None  # if not None, will override the number of pages

    # 懒加载并缓存 HF 模型配置（首次访问时读盘）
    @cached_property
    def hf_config(self):
        # 从模型路径加载 config.json
        return cached_load_hf_config(self.model_path)

    # 懒加载并缓存引擎侧的模型配置（层数/头数/rotary 等）
    @cached_property
    def model_config(self) -> ModelConfig:
        # 运行时导入避免循环依赖
        from minisgl.models import ModelConfig

        # 从 HF 配置转换
        return ModelConfig.from_hf(self.hf_config)

    # 引擎实际采用的最大序列长度：用户覆盖优先，否则用 rotary 的 max_position
    @property
    def max_seq_len(self) -> int:
        # 有覆盖值就用覆盖值
        if self.max_seq_len_override is not None:
            # 返回覆盖值
            return self.max_seq_len_override
        # 默认取 rotary 配置的最大位置数
        return self.model_config.rotary_config.max_position

    # 单次 forward 允许处理的最大 token 长度（当前等于 max_seq_len）
    @property
    def max_forward_len(self) -> int:
        # 直接沿用最大序列长度
        return self.max_seq_len

    # TP 进程组 bootstrap 用的 TCP 地址（本机端口 2333）
    @property
    def distributed_addr(self) -> str:
        # 固定本机地址
        return "tcp://127.0.0.1:2333"
