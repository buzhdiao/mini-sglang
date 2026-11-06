# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# argparse：命令行参数解析
import argparse
# os：路径展开（~）等
import os
# dataclass：生成配置类
from dataclasses import dataclass
# 类型注解工具
from typing import List, Tuple

import torch
# TP 并行拓扑信息
from minisgl.distributed import DistributedInfo
# 调度器配置基类（ServerArgs 继承它）
from minisgl.scheduler import SchedulerConfig
# 日志器
from minisgl.utils import init_logger


# 服务器级配置：在调度器配置之上追加 HTTP/进程拓扑等启动参数（不可变）
@dataclass(frozen=True)
class ServerArgs(SchedulerConfig):
    # HTTP 服务监听地址（默认本机回环）
    server_host: str = "127.0.0.1"
    # HTTP 服务监听端口
    server_port: int = 1919
    # tokenizer 子进程数量：0 表示与 detokenizer 合并成一个进程
    num_tokenizer: int = 0
    # 静默模式：关闭 INFO 级日志（shell 模式下保持终端干净）
    silent_output: bool = False

    @property
    def share_tokenizer(self) -> bool:
        # 未单独启动 tokenizer 进程即"共享"模式（detokenizer 顺带做分词）
        return self.num_tokenizer == 0

    @property
    def zmq_frontend_addr(self) -> str:
        # 前端(API server)接收回包的 IPC 地址（tokenizer -> frontend）
        return "ipc:///tmp/minisgl_3" + self._unique_suffix

    @property
    def zmq_tokenizer_addr(self) -> str:
        if self.share_tokenizer:
            # 共享模式：没有独立 tokenizer，前端直连 detokenizer 的收包地址
            return self.zmq_detokenizer_addr
        # 独立 tokenizer 模式：前端把请求推到专用地址
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix
        # 防御性检查：两条地址绝不能相同（否则消息串线）
        assert result != self.zmq_detokenizer_addr
        # 返回专用地址
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        # 共享模式下由 detokenizer（即 tokenizer 进程）负责创建收包端点
        return self.share_tokenizer

    @property
    def backend_create_detokenizer_link(self) -> bool:
        # 独立 tokenizer 模式下调度器不再直连 detokenizer（链路由 detokenizer 建）
        return not self.share_tokenizer

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        # 独立 tokenizer 模式下前端负责创建发往 tokenizer 的端点
        return not self.share_tokenizer

    @property
    def distributed_addr(self) -> str:
        # 分布式初始化（torch.distributed store）用的 TCP 地址：端口+1 避开 HTTP 端口
        return f"tcp://127.0.0.1:{self.server_port + 1}"


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.

    Args:
        args: Command line arguments (e.g., sys.argv[1:])

    Returns:
        EngineConfig instance with parsed arguments
    """
    # 函数内导入：延迟加载重型模块，缩短 --help 响应时间
    from minisgl.attention import validate_attn_backend
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER
    from minisgl.moe import SUPPORTED_MOE_BACKENDS

    # 创建参数解析器
    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")

    parser.add_argument(
        # 模型路径（可长写或短写）
        "--model-path",
        "--model",
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
    )

    parser.add_argument(
        # 权重精度
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",
    )

    parser.add_argument(
        # TP 并行度
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",
    )

    parser.add_argument(
        # 最大并发请求数（决定页表行数）
        "--max-running-requests",
        type=int,
        dest="max_running_req",
        default=ServerArgs.max_running_req,
        help="The maximum number of running requests.",
    )

    parser.add_argument(
        # 手动覆盖最大序列长度（默认从模型配置读取）
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,
        help="The maximum sequence length override.",
    )

    parser.add_argument(
        # KV cache 可用的显存比例
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,
        help="The fraction of GPU memory to use for KV cache.",
    )

    # 断言 dataclass 默认值与期望一致（防止上游改动后此处语义失效）
    assert ServerArgs.use_dummy_weight == False
    parser.add_argument(
        # 使用假权重（跳过权重加载，用于测试调度/通信路径）
        "--dummy-weight",
        action="store_true",
        dest="use_dummy_weight",
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True
    parser.add_argument(
        # 关闭 PyNCCL（退回 torch.distributed 原生集合通信）
        "--disable-pynccl",
        action="store_false",
        dest="use_pynccl",
        help="Disable PyNCCL for tensor parallelism.",
    )

    parser.add_argument(
        # HTTP 监听地址
        "--host",
        type=str,
        dest="server_host",
        default=ServerArgs.server_host,
        help="The host address for the server.",
    )

    parser.add_argument(
        # HTTP 监听端口
        "--port",
        type=int,
        dest="server_port",
        default=ServerArgs.server_port,
        help="The port number for the server to listen on.",
    )

    parser.add_argument(
        # CUDA Graph 捕获的最大批量（决定显存占用与可回放的批上限）
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(
        # tokenizer 子进程数量（0 表示与 detokenizer 合并）
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",
    )

    parser.add_argument(
        # chunked prefill 的分块大小（单次 forward 最大扩展 token 数）
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",
        default=ServerArgs.max_extend_tokens,
        help="Chunk Prefill maximum chunk size in tokens.",
    )

    parser.add_argument(
        # 手动指定 KV cache 页数上限（默认按显存自动推算）
        "--num-pages",
        dest="num_page_override",
        type=int,
        default=ServerArgs.num_page_override,
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(
        # KV cache 页大小（1 表示逐 token 粒度管理）
        "--page-size",
        type=int,
        default=ServerArgs.page_size,
        help="Set the page size for system management.",
    )

    parser.add_argument(
        # 注意力后端选择（可给两个：prefill 用第一个，decode 用第二个）
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,
        default=ServerArgs.attention_backend,
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",
    )

    parser.add_argument(
        # 模型下载源
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(
        # KV cache 管理策略（radix 等）
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),
        help="The KV cache management strategy.",
    )

    parser.add_argument(
        # MoE 后端选择（auto 表示按模型/硬件自动挑）
        "--moe-backend",
        default=ServerArgs.moe_backend,
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),
        help="The MoE backend to use.",
    )

    parser.add_argument(
        # 以交互式 shell 模式运行（单请求、无 HTTP）
        "--shell-mode",
        action="store_true",
        help="Run the server in shell mode.",
    )

    # Parse arguments
    # 解析命令行并转成普通字典（复制一份避免修改 Namespace 内部）
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    # 取出 shell_mode 标志（从 kwargs 移除，因其不是 ServerArgs 字段）
    run_shell |= kwargs.pop("shell_mode")
    if run_shell:
        # shell 模式：批量=1（单请求交互）、最大并发=1、静默日志
        kwargs["cuda_graph_max_bs"] = 1
        kwargs["max_running_req"] = 1
        kwargs["silent_output"] = True

    if kwargs["model_path"].startswith("~"):
        # 展开 ~ 为用户主目录
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":
        # ModelScope 源：本地不存在时先下载快照
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):
            # 延迟导入 modelscope（重型依赖）
            from modelscope import snapshot_download

            # dummy 模式：跳过下载权重文件本身（只需要配置/tokenizer 文件）
            ignore_patterns = []
            if kwargs["use_dummy_weight"]:
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            # 下载快照并拿到本地路径
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)
            kwargs["model_path"] = model_path
    # model_source 只是解析期辅助，不进入配置
    del kwargs["model_source"]

    if (dtype_str := kwargs["dtype"]) == "auto":
        # auto 模式：读 HF 配置里的 dtype 字段（带缓存，重复读不重复下载/解析）
        from minisgl.utils import cached_load_hf_config

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype

    # 字符串 dtype -> torch dtype 的映射表
    DTYPE_MAP = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    # 字符串则查表转换；本身已是 torch dtype 则原样保留
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str
    # 把 TP 规模换成 DistributedInfo 对象（rank0 的视角）
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])
    # tensor_parallel_size 已被消化，不进入配置
    del kwargs["tensor_parallel_size"]

    # 用清洗后的字段构造不可变配置
    result = ServerArgs(**kwargs)
    # 初始化本模块日志器
    logger = init_logger(__name__)
    # 打印最终生效的配置
    logger.info(f"Parsed arguments:\n{result}")
    # 返回配置与是否 shell 模式
    return result, run_shell
