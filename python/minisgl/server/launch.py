# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# logging：关停时临时禁用 INFO 日志
import logging
# multiprocessing：启动调度器/tokenizer 子进程
import multiprocessing as mp
# sys：读取命令行参数
import sys
# replace：以字段覆盖方式派生不可变 dataclass 副本
from dataclasses import replace
# 类型注解工具
from typing import TYPE_CHECKING

# TP 拓扑信息
from minisgl.distributed import DistributedInfo
# 日志器
from minisgl.utils import init_logger

# 仅类型检查导入，避免运行时循环依赖
if TYPE_CHECKING:
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    # 子进程入口：函数内导入，spawn 后重新加载必要模块
    import torch
    from minisgl.scheduler import Scheduler

    # 全程推理模式：禁用梯度跟踪，省显存提性能
    with torch.inference_mode():
        # 构造调度器（内含引擎初始化：模型加载、KV 池、CUDA Graph）
        scheduler = Scheduler(args)
        # 全 rank 屏障：确保所有 TP rank 的引擎都初始化完毕
        scheduler.sync_all_ranks()

        if args.tp_info.is_primary():
            # 只有 rank0 向主进程回报就绪（避免重复 ack）
            ack_queue.put("Scheduler is ready")

        if args.silent_output:
            # 静默模式：全局关闭 INFO 日志（运行期日志也不打）
            logging.disable(logging.INFO)

        try:
            # 进入调度主循环（永不返回）
            scheduler.run_forever()
        except KeyboardInterrupt:
            # 收到退出消息（ExitMsg 转 KeyboardInterrupt）或 Ctrl-C：优雅退出
            logger = init_logger(__name__)
            if args.tp_info.is_primary():
                # ^C 后补一个换行让日志不贴在提示符后
                print()  # for a clean newline after ^C
                logger.info("Scheduler exiting gracefully...")
            # 释放引擎资源、同步各 rank 后退出
            scheduler.shutdown()


def launch_server(run_shell: bool = False) -> None:
    # 函数内导入：延迟加载服务端模块
    from .api_server import run_api_server
    from .args import parse_args

    # 解析命令行得到服务器配置与 shell 模式标志
    server_args, run_shell = parse_args(sys.argv[1:], run_shell)
    # 初始化日志器（固定名称 initializer）
    logger = init_logger(__name__, "initializer")

    def start_subprocess() -> None:
        # 启动全部后端子进程（在 API server 初始化后被回调）
        import multiprocessing as mp

        from minisgl.tokenizer import tokenize_worker

        # 强制 spawn 启动方式：子进程全新解释器，CUDA/线程状态干净
        mp.set_start_method("spawn", force=True)

        # TP 并行度（调度器子进程数量）
        world_size = server_args.tp_info.size
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        # 就绪回报队列：每个子进程初始化完成后放入一条确认消息
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):
            # 为每个 TP rank 派生一份配置（只改 tp_info 字段）
            new_args = replace(
                server_args,
                tp_info=DistributedInfo(i, world_size),
            )
            mp.Process(
                # 调度器子进程入口
                target=_run_scheduler,
                args=(new_args, ack_queue),
                # daemon=False：主进程退出不会强杀调度器（保证优雅退出路径）
                daemon=False,
                name=f"minisgl-TP{i}-scheduler",
            ).start()

        # 独立 tokenizer 进程数量
        num_tokenizers = server_args.num_tokenizer
        # DeTokenizer, only 1
        # 反解码进程固定 1 个：收调度器 token 流，反解码后发给前端
        mp.Process(
            target=tokenize_worker,
            kwargs={
                # tokenizer 权重/词表路径
                "tokenizer_path": server_args.model_path,
                # 本进程的收包地址（调度器把 token 推到这里）
                "addr": server_args.zmq_detokenizer_addr,
                # 调度器的收包地址（分词请求/回包都经调度器）
                "backend_addr": server_args.zmq_backend_addr,
                # 前端的收包地址（最终结果发给 API server）
                "frontend_addr": server_args.zmq_frontend_addr,
                # 本进程一次处理的请求数上限
                "local_bs": 1,
                # 是否由本进程创建收包端点（共享模式为 True）
                "create": server_args.tokenizer_create_addr,
                # 进程编号（反解码进程编号 = num_tokenizers）
                "tokenizer_id": num_tokenizers,
                "ack_queue": ack_queue,
            },
            daemon=False,
            name="minisgl-detokenizer-0",
        ).start()
        for i in range(num_tokenizers):
            mp.Process(
                # 独立 tokenizer 进程：负责把文本转 token 发给调度器
                target=tokenize_worker,
                kwargs={
                    "tokenizer_path": server_args.model_path,
                    "addr": server_args.zmq_tokenizer_addr,
                    "backend_addr": server_args.zmq_backend_addr,
                    "frontend_addr": server_args.zmq_frontend_addr,
                    "local_bs": 1,
                    "create": server_args.tokenizer_create_addr,
                    "tokenizer_id": i,
                    "ack_queue": ack_queue,
                },
                daemon=False,
                name=f"minisgl-tokenizer-{i}",
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        # 逐条收取全部就绪确认（get 阻塞直到子进程回报），
        # 确保后端完全就绪后 API server 才开始对外服务
        for _ in range(num_tokenizers + 2):
            logger.info(ack_queue.get())

    # 启动 API server（内部回调 start_subprocess 拉起后端）
    run_api_server(server_args, start_subprocess, run_shell=run_shell)


if __name__ == "__main__":
    # 直接运行本模块即启动服务器
    launch_server()
