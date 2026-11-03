from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解使用前向引用

import argparse  # 命令行参数解析库
import os  # 文件路径处理（~ 展开、目录判断）
from dataclasses import dataclass  # dataclass 装饰器
from typing import List, Tuple  # 类型注解

import torch  # PyTorch：dtype 字符串到 torch 数据类型的映射
from minisgl.distributed import DistributedInfo  # 分布式信息（rank/world_size），用于构造 tp_info
from minisgl.scheduler import SchedulerConfig  # 调度器配置基类：ServerArgs 继承其全部调度相关字段
from minisgl.utils import init_logger  # 日志初始化


@dataclass(frozen=True)  # 不可变数据类：创建后字段不可修改（frozen 保证配置安全，但可用 dataclasses.replace 派生新实例）
class ServerArgs(SchedulerConfig):
    """服务器级配置：在调度器配置基础上追加 HTTP 服务与 ZMQ 拓扑相关字段"""

    server_host: str = "127.0.0.1"  # HTTP 服务监听地址
    server_port: int = 1919  # HTTP 服务监听端口
    num_tokenizer: int = 0  # 独立 tokenizer 进程数；0 表示与 detokenizer 合并成一个进程（共享模式）
    silent_output: bool = False  # 是否静默输出（shell 模式下设为 True，避免日志刷屏）

    @property  # 属性装饰器：像字段一样访问的计算属性
    def share_tokenizer(self) -> bool:
        """是否为 tokenizer/detokenizer 共享模式（num_tokenizer==0 时二者合并为一个进程）"""
        return self.num_tokenizer == 0  # 没有独立 tokenizer 即共享

    @property
    def zmq_frontend_addr(self) -> str:
        """前端（API server）接收回复的 ZMQ IPC 地址（tokenizer/detokenizer 把结果推到这里）"""
        return "ipc:///tmp/minisgl_3" + self._unique_suffix  # 用唯一后缀避免同机多实例冲突（继承自 SchedulerConfig）

    @property
    def zmq_tokenizer_addr(self) -> str:
        """tokenizer 接收请求的 ZMQ IPC 地址（前端把编码请求推到这里）"""
        if self.share_tokenizer:  # 共享模式：tokenizer 与 detokenizer 是同一进程，直接复用 detokenizer 地址
            return self.zmq_detokenizer_addr
        result = "ipc:///tmp/minisgl_4" + self._unique_suffix  # 独立模式：使用专属地址（编号 4，与 detokenizer 的 2/3 区分开）
        assert result != self.zmq_detokenizer_addr  # 防御性断言：两地址必须不同，否则消息路由会错乱
        return result

    @property
    def tokenizer_create_addr(self) -> bool:
        """tokenizer 进程是否负责 bind 自己的 ZMQ 地址（True=共享模式下由该进程 bind）"""
        return self.share_tokenizer  # 共享模式下该进程要创建链路；独立模式下由别人 bind

    @property
    def backend_create_detokenizer_link(self) -> bool:
        """调度器（后端）是否负责 bind detokenizer 链路（独立 tokenizer 模式下 detokenizer 地址归后端管）"""
        return not self.share_tokenizer  # 独立模式：后端 bind detokenizer 地址，共享模式则由 detokenizer 自己 bind

    @property
    def frontend_create_tokenizer_link(self) -> bool:
        """前端是否负责 bind tokenizer 链路（独立 tokenizer 模式下地址归前端管）"""
        return not self.share_tokenizer  # 独立模式：前端 bind tokenizer 地址；共享模式下 tokenizer 自己 bind

    @property
    def distributed_addr(self) -> str:
        """进程组初始化（torch.distributed rendezvous）用的 TCP 地址，端口复用 server_port+1"""
        return f"tcp://127.0.0.1:{self.server_port + 1}"  # 基于 HTTP 端口 +1，天然避免与 HTTP/其他 ZMQ 端口冲突


def parse_args(args: List[str], run_shell: bool = False) -> Tuple[ServerArgs, bool]:
    """
    Parse command line arguments and return an EngineConfig.
    解析命令行参数并返回 EngineConfig（此处实际返回 ServerArgs）。

    Args:
        args: Command line arguments (e.g., sys.argv[1:])
            命令行参数列表（如 sys.argv[1:]）
        run_shell: 是否已由外部指定进入 shell 模式

    Returns:
        EngineConfig instance with parsed arguments
        解析完成（含类型转换/默认值/派生处理）的 ServerArgs 与最终 run_shell 标志
    """
    from minisgl.attention import validate_attn_backend  # 延迟导入：注意力后端名校验函数（作为 --attention-backend 的 type）
    from minisgl.kvcache import SUPPORTED_CACHE_MANAGER  # 延迟导入：支持的 KV cache 管理器注册表（用于 choices）
    from minisgl.moe import SUPPORTED_MOE_BACKENDS  # 延迟导入：支持的 MoE 后端注册表（用于 choices）
    # 以上延迟到函数内导入：避免 import 本模块时牵连加载重依赖（torch/flashinfer 等）

    parser = argparse.ArgumentParser(description="MiniSGL Server Arguments")  # 创建参数解析器

    parser.add_argument(  # --model-path / --model：必填
        "--model-path",  # 标准选项名
        "--model",  # 短别名
        type=str,
        required=True,
        help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",  # 权重路径：本地目录或 HF repo id
    )

    parser.add_argument(  # --dtype：权重/激活精度
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights and activations. 'auto' will use FP16 for FP32/FP16 models and BF16 for BF16 models.",  # auto：按模型配置自动选 FP16/BF16
    )

    parser.add_argument(  # --tensor-parallel-size / --tp-size：张量并行度
        "--tensor-parallel-size",
        "--tp-size",
        type=int,
        default=1,
        help="The tensor parallelism size.",  # 张量并行规模（会启动等量调度器进程）
    )

    parser.add_argument(  # --max-running-requests：最大并发请求数
        "--max-running-requests",
        type=int,
        dest="max_running_req",  # 映射到 ServerArgs.max_running_req 字段
        default=ServerArgs.max_running_req,  # 默认值取自父类 SchedulerConfig 的字段默认值
        help="The maximum number of running requests.",
    )

    parser.add_argument(  # --max-seq-len-override：覆盖模型声明的最大序列长度
        "--max-seq-len-override",
        type=int,
        default=ServerArgs.max_seq_len_override,  # 默认取父类字段值（None 表示不覆盖）
        help="The maximum sequence length override.",
    )

    parser.add_argument(  # --memory-ratio：KV cache 显存占比
        "--memory-ratio",
        type=float,
        default=ServerArgs.memory_ratio,  # 默认取父类字段值
        help="The fraction of GPU memory to use for KV cache.",  # GPU 显存中用于 KV cache 的比例
    )

    assert ServerArgs.use_dummy_weight == False  # 编译期约束：父类默认必须是 False（该选项只能打开，不能关）
    parser.add_argument(  # --dummy-weight：使用随机权重（功能测试用）
        "--dummy-weight",
        action="store_true",  # 出现该选项即置 True，不出现保持 False
        dest="use_dummy_weight",  # 映射到字段 use_dummy_weight
        help="Use dummy weights for testing.",
    )

    assert ServerArgs.use_pynccl == True  # 编译期约束：父类默认必须是 True（该选项只能关闭，不能开）
    parser.add_argument(  # --disable-pynccl：禁用 PyNCCL 通信后端
        "--disable-pynccl",
        action="store_false",  # 出现该选项即置 False
        dest="use_pynccl",  # 映射到字段 use_pynccl
        help="Disable PyNCCL for tensor parallelism.",  # TP 张量并行通信回退到非 pynccl 路径
    )

    parser.add_argument(  # --host：监听地址
        "--host",
        type=str,
        dest="server_host",  # 映射到字段 server_host
        default=ServerArgs.server_host,  # 默认 127.0.0.1
        help="The host address for the server.",
    )

    parser.add_argument(  # --port：监听端口
        "--port",
        type=int,
        dest="server_port",  # 映射到字段 server_port
        default=ServerArgs.server_port,  # 默认 1919
        help="The port number for the server to listen on.",
    )

    parser.add_argument(  # --cuda-graph-max-bs / --graph：CUDA Graph 捕获的最大 batch size
        "--cuda-graph-max-bs",
        "--graph",
        type=int,
        default=ServerArgs.cuda_graph_max_bs,  # 默认取父类字段值（None 表示按显存自动调优）
        help="The maximum batch size for CUDA graph capture. None means auto-tuning based on the GPU memory.",
    )

    parser.add_argument(  # --num-tokenizer / --tokenizer-count：独立 tokenizer 进程数
        "--num-tokenizer",
        "--tokenizer-count",
        type=int,
        default=ServerArgs.num_tokenizer,  # 默认 0（共享模式）
        help="The number of tokenizer processes to launch. 0 means the tokenizer is shared with the detokenizer.",  # 0 表示与 detokenizer 合并
    )

    parser.add_argument(  # --max-prefill-length / --max-extend-length：chunk prefill 单块最大 token 数
        "--max-prefill-length",
        "--max-extend-length",
        type=int,
        dest="max_extend_tokens",  # 映射到父类字段 max_extend_tokens
        default=ServerArgs.max_extend_tokens,  # 默认取父类字段值
        help="Chunk Prefill maximum chunk size in tokens.",  # 长 prompt 会被切块 prefill，此为每块上限
    )

    parser.add_argument(  # --num-pages：KV cache 页数上限覆盖
        "--num-pages",
        dest="num_page_override",  # 映射到父类字段 num_page_override
        type=int,
        default=ServerArgs.num_page_override,  # 默认取父类字段值（None 表示按显存自动计算）
        help="Set the maximum number of pages for KVCache.",
    )

    parser.add_argument(  # --page-size：KV cache 页大小
        "--page-size",
        type=int,
        default=ServerArgs.page_size,  # 默认取父类字段值
        help="Set the page size for system management.",  # PagedAttention 每页容纳的 token 数
    )

    parser.add_argument(  # --attention-backend / --attn：注意力后端选择
        "--attention-backend",
        "--attn",
        type=validate_attn_backend,  # 用校验函数做类型转换（支持 "a" 或 "a,b" 分别指定 prefill/decode 后端）
        default=ServerArgs.attention_backend,  # 默认取父类字段值
        help="The attention backend to use. If two backends are specified,"
        " the first one is used for prefill and the second one for decode.",  # 可指定两个：第一个 prefill，第二个 decode
    )

    parser.add_argument(  # --model-source：模型下载源
        "--model-source",
        type=str,
        default="huggingface",
        choices=["huggingface", "modelscope"],  # 仅支持 HF 与魔搭两种来源
        help="The source to download model from. Either 'huggingface' or 'modelscope'.",
    )

    parser.add_argument(  # --cache-type：KV cache 管理策略
        "--cache-type",
        type=str,
        default=ServerArgs.cache_type,  # 默认取父类字段值
        choices=SUPPORTED_CACHE_MANAGER.supported_names(),  # 可选值来自注册表动态生成
        help="The KV cache management strategy.",
    )

    parser.add_argument(  # --moe-backend：MoE 后端选择
        "--moe-backend",
        default=ServerArgs.moe_backend,  # 默认取父类字段值
        choices=["auto"] + SUPPORTED_MOE_BACKENDS.supported_names(),  # auto + 注册表中已支持的后端
        help="The MoE backend to use.",
    )

    parser.add_argument(  # --shell-mode：交互式终端模式
        "--shell-mode",
        action="store_true",  # 出现即置 True
        help="Run the server in shell mode.",  # 不起 HTTP 服务，改为本地交互对话
    )

    # Parse arguments
    # 解析参数：把 Namespace 转成 dict 拷贝（后续会增删键）
    kwargs = parser.parse_args(args).__dict__.copy()

    # resolve some arguments
    # 派生/修正参数
    run_shell |= kwargs.pop("shell_mode")  # 取出 shell_mode 并与外部传入的 run_shell 取或
    if run_shell:  # shell 模式强制单请求、batch=1、静默输出
        kwargs["cuda_graph_max_bs"] = 1  # 只捕获 bs=1 的 CUDA Graph 即可
        kwargs["max_running_req"] = 1  # 交互模式同时只处理一个请求
        kwargs["silent_output"] = True  # 不在终端刷日志（终端被对话输出占用）

    if kwargs["model_path"].startswith("~"):  # 模型路径以 ~ 开头：展开为用户主目录绝对路径
        kwargs["model_path"] = os.path.expanduser(kwargs["model_path"])

    if kwargs["model_source"] == "modelscope":  # 魔搭来源：本地不存在时先下载
        model_path = kwargs["model_path"]
        if not os.path.isdir(model_path):  # 非本地目录则视为魔搭 repo id，需要下载
            from modelscope import snapshot_download  # 延迟导入：只有魔搭路径才需要该依赖

            ignore_patterns = []  # 默认下载全部文件
            if kwargs["use_dummy_weight"]:  # 假权重模式：跳过真正的权重文件下载（省时间/带宽）
                ignore_patterns = ["*.bin", "*.safetensors", "*.pt", "*.ckpt"]
            model_path = snapshot_download(model_path, ignore_patterns=ignore_patterns)  # 全量快照下载，返回本地缓存路径
            kwargs["model_path"] = model_path  # 用下载后的本地路径替换 repo id
    del kwargs["model_source"]  # 该字段不属于 ServerArgs（或已消费），从 kwargs 中移除

    if (dtype_str := kwargs["dtype"]) == "auto":  # 海象表达式：取出 dtype 并判断是否 auto
        from minisgl.utils import cached_load_hf_config  # 延迟导入：带缓存的 HF config 加载器

        dtype_str = cached_load_hf_config(kwargs["model_path"]).dtype  # 从模型 config.json 读取声明的精度（如 float16/bfloat16）

    DTYPE_MAP = {  # 字符串 → torch dtype 映射表
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    kwargs["dtype"] = DTYPE_MAP[dtype_str] if isinstance(dtype_str, str) else dtype_str  # 字符串转 torch dtype；非字符串（异常路径）原样保留
    kwargs["tp_info"] = DistributedInfo(0, kwargs["tensor_parallel_size"])  # 构造 rank=0 的分布式信息（子进程会被 replace 成各自 rank）
    del kwargs["tensor_parallel_size"]  # 该字段已转成 tp_info，从 kwargs 移除（ServerArgs 无此字段）

    result = ServerArgs(**kwargs)  # 用处理后的 kwargs 构造不可变配置对象（字段不匹配会在这里报错）
    logger = init_logger(__name__)  # 初始化日志器
    logger.info(f"Parsed arguments:\n{result}")  # 打印最终解析结果（dataclass repr）
    return result, run_shell  # 返回配置与最终 shell 标志
