from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解使用前向引用

import logging  # 日志级别控制（静默模式下禁用 INFO 日志）
import multiprocessing as mp  # 多进程：拉起调度器子进程
import sys  # 读取命令行参数 sys.argv
from dataclasses import replace  # 对 frozen dataclass 派生修改副本（给每个 TP rank 生成独立 tp_info）
from typing import TYPE_CHECKING  # 类型检查专用导入开关（避免运行时循环导入）

from minisgl.distributed import DistributedInfo  # 分布式信息（rank/world_size）
from minisgl.utils import init_logger  # 日志初始化

if TYPE_CHECKING:  # 仅在静态类型检查时导入，运行时不执行（避免循环导入）
    from .args import ServerArgs


def _run_scheduler(args: ServerArgs, ack_queue: mp.Queue[str]) -> None:
    """调度器子进程入口：初始化 Scheduler 并进入事件循环，就绪后向主进程发 ack"""
    import torch  # 延迟导入：子进程（spawn）启动后才加载，隔离主进程与重依赖
    from minisgl.scheduler import Scheduler  # 延迟导入：调度器主类

    with torch.inference_mode():  # 推理模式：整个调度器生命周期禁用梯度，省内存提速度
        scheduler = Scheduler(args)  # 构造调度器（加载模型权重、初始化 KV cache、CUDA Graph、ZMQ 链路等重活）
        scheduler.sync_all_ranks()  # 屏障同步：所有 TP rank 都完成初始化后才继续（保证通信组一致）

        if args.tp_info.is_primary():  # 只有主 rank（rank 0）发就绪确认，避免 ack 数量随 TP 翻倍
            ack_queue.put("Scheduler is ready")  # 通知主进程：本组调度器已就绪

        if args.silent_output:  # 静默模式（shell 模式）
            logging.disable(logging.INFO)  # 全局禁用 INFO 及以下日志，避免刷屏干扰对话输出

        try:
            scheduler.run_forever()  # 进入调度主循环（prefill/decode 轮转），直到收到中断
        except KeyboardInterrupt:  # Ctrl-C 触发（多进程下通常由主进程先收到再传播）
            logger = init_logger(__name__)  # 退出阶段才初始化日志器
            if args.tp_info.is_primary():  # 主 rank 打印退出信息
                print()  # for a clean newline after ^C  # ^C 后补一个换行，让日志不接在提示符后
                logger.info("Scheduler exiting gracefully...")  # 优雅退出日志
            scheduler.shutdown()  # 执行清理（释放资源、关闭 ZMQ socket 等）


def launch_server(run_shell: bool = False) -> None:
    """服务器启动总入口：解析参数 → 拉起全部子进程 → 启动前端 API server（或 shell）"""
    from .api_server import run_api_server  # 延迟导入：前端 API server 启动函数
    from .args import parse_args  # 延迟导入：命令行参数解析

    server_args, run_shell = parse_args(sys.argv[1:], run_shell)  # 解析命令行参数，得到配置与最终 shell 标志
    logger = init_logger(__name__, "initializer")  # 初始化启动器日志器

    def start_subprocess() -> None:
        """拉起所有后端子进程：TP 调度器 ×N + detokenizer ×1 + tokenizer ×M"""
        import multiprocessing as mp  # 延迟导入：仅在需要拉起子进程时加载

        from minisgl.tokenizer import tokenize_worker  # 延迟导入：tokenizer/detokenizer worker 入口

        mp.set_start_method("spawn", force=True)  # 强制 spawn 启动方式：子进程是全新解释器，规避 CUDA 与 fork 的兼容问题

        world_size = server_args.tp_info.size  # TP 世界大小（调度器进程数）
        # a multiprocessing queue to receive ack from subprocesses
        # so that we can guarantee all subprocesses are ready
        # 就绪确认队列：主进程阻塞收集所有子进程的 ack，确保全部就绪后才开始对外服务
        ack_queue: mp.Queue[str] = mp.Queue()

        for i in range(world_size):  # 逐个启动 TP 调度器进程
            new_args = replace(  # frozen dataclass 不可原地改，用 replace 派生新配置
                server_args,
                tp_info=DistributedInfo(i, world_size),  # 仅替换 tp_info：每个进程拿到自己的 rank
            )
            mp.Process(  # 创建调度器子进程
                target=_run_scheduler,  # 入口函数
                args=(new_args, ack_queue),  # 传入本 rank 配置与 ack 队列
                daemon=False,  # 非守护进程：主进程退出不会强杀（配合 psutil 主动清理）
                name=f"minisgl-TP{i}-scheduler",  # 进程名：便于 ps/日志识别
            ).start()

        num_tokenizers = server_args.num_tokenizer  # 独立 tokenizer 进程数
        # DeTokenizer, only 1
        # detokenizer：固定 1 个进程（token id → 文本 的增量解码在这里做）
        mp.Process(
            target=tokenize_worker,  # 与 tokenizer 共用同一入口函数（靠 addr/参数区分角色）
            kwargs={
                "tokenizer_path": server_args.model_path,  # 分词器路径（与模型同目录）
                "addr": server_args.zmq_detokenizer_addr,  # detokenizer 的接收地址（调度器把新 token 推到这里）
                "backend_addr": server_args.zmq_backend_addr,  # 回发调度器的地址（共享模式下也接收前端请求）
                "frontend_addr": server_args.zmq_frontend_addr,  # 推送结果给前端的地址
                "local_bs": 1,  # 本地攒批大小为 1（低延迟优先）
                "create": server_args.tokenizer_create_addr,  # 是否由本进程 bind 地址（共享模式下为 True）
                "tokenizer_id": num_tokenizers,  # 用 num_tokenizers 作为 id，日志显示为 detokenizer
                "ack_queue": ack_queue,  # 就绪确认队列
            },
            daemon=False,  # 非守护进程
            name="minisgl-detokenizer-0",  # 进程名：detokenizer-0
        ).start()
        for i in range(num_tokenizers):  # 启动独立 tokenizer 进程（共享模式下 num_tokenizers=0，循环不执行）
            mp.Process(
                target=tokenize_worker,  # 同一入口：处理前端请求的文本编码
                kwargs={
                    "tokenizer_path": server_args.model_path,  # 分词器路径
                    "addr": server_args.zmq_tokenizer_addr,  # tokenizer 的接收地址（前端把编码请求推到这里）
                    "backend_addr": server_args.zmq_backend_addr,  # 把 input_ids 推给调度器的地址
                    "frontend_addr": server_args.zmq_frontend_addr,  # 推送结果给前端的地址（独立模式基本不用）
                    "local_bs": 1,  # 本地攒批大小为 1
                    "create": server_args.tokenizer_create_addr,  # 独立模式下为 False（地址由前端 bind）
                    "tokenizer_id": i,  # 实例编号（日志区分）
                    "ack_queue": ack_queue,  # 就绪确认队列
                },
                daemon=False,  # 非守护进程
                name=f"minisgl-tokenizer-{i}",  # 进程名：tokenizer-i
            ).start()

        # Wait for acknowledgments from all worker processes:
        # - world_size schedulers (but only primary rank sends ack)
        # - num_tokenizers tokenizers
        # - 1 detokenizer
        # Total acks expected: 1 + num_tokenizers + 1 = num_tokenizers + 2
        # 等待所有子进程的 ack：TP 组只有主 rank 发 1 条，加 N 个 tokenizer 和 1 个 detokenizer，共 N+2 条
        for _ in range(num_tokenizers + 2):  # 阻塞收齐全部 ack
            logger.info(ack_queue.get())  # 逐条打印就绪消息（get 无 ack 时会阻塞）

    run_api_server(server_args, start_subprocess, run_shell=run_shell)  # 启动前端：先执行 start_subprocess 再起 HTTP/shell


if __name__ == "__main__":  # 直接运行本文件时启动服务器（python -m minisgl.server.launch）
    launch_server()
