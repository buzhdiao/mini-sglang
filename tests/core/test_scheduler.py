from __future__ import annotations  # 启用 PEP 563 延迟注解求值

import torch  # 导入 torch，用于张量拼接、dtype 转换与推理模式
import multiprocessing as mp  # 导入多进程模块，把 Scheduler 跑在独立子进程中
from transformers import AutoTokenizer  # 导入分词器，编码 prompt 与解码输出

from minisgl.distributed import DistributedInfo  # 导入分布式信息结构（TP rank/size）
from minisgl.message import BaseBackendMsg, BaseTokenizerMsg, DetokenizeMsg, ExitMsg, UserMsg  # 导入进程间消息：编解码基类与具体消息类型
from minisgl.scheduler import Scheduler, SchedulerConfig  # 导入被测的调度器及其配置类
from minisgl.utils import ZmqPullQueue, ZmqPushQueue, call_if_main, init_logger  # 导入 ZMQ 收发队列与工具装饰器/日志
from minisgl.core import SamplingParams  # 导入采样参数配置

logger = init_logger(__name__)  # 创建本模块 logger


@torch.inference_mode()  # 推理模式：禁用 autograd 追踪，降低显存与计算开销
def scheduler(config: SchedulerConfig, queue: mp.Queue) -> None:
    """调度器子进程入口：初始化 Scheduler，就绪后通知主进程，然后进入永久调度循环。"""
    scheduler = Scheduler(config)  # 创建调度器（加载模型、初始化 KV cache/各组件，耗时较长）
    queue.put(None)  # 向主进程发送就绪信号（内容为 None 占位即可）
    try:
        scheduler.run_forever()  # 进入永久调度循环（持续处理请求直到被中断）
    except KeyboardInterrupt:  # 收到键盘中断（Ctrl+C）时
        logger.info_rank0("Scheduler exiting...")  # 打印退出日志（仅 rank0 打印）


@call_if_main(__name__)  # 装饰器：仅在本文件作为主模块运行时执行函数体（被 import 时不执行）
def main():
    """集成测试主流程：子进程启动 Scheduler，主进程模拟 detokenizer 角色，收发消息完成一次完整生成。"""
    config = SchedulerConfig(
        model_path="meta-llama/Llama-3.1-8B-Instruct",  # 被测模型路径（需提前下载或可访问）
        tp_info=DistributedInfo(0, 1),  # 张量并行信息：单卡（rank 0 / size 1）
        dtype=torch.bfloat16,  # 权重精度 bfloat16
        max_running_req=4,  # 最大并发请求数 4
        cuda_graph_bs=[2, 4, 8],  # 预捕获 CUDA Graph 的 batch size 档位
    )

    mp.set_start_method("spawn", force=True)  # 使用 spawn 启动方式（CUDA 子进程必需），force 覆盖已有设置
    q = mp.Queue()  # 进程间队列：专门传递就绪信号
    p = mp.Process(target=scheduler, args=(config, q))  # 创建调度器子进程
    p.start()  # 启动子进程
    q.get()  # 阻塞等待子进程就绪信号（模型加载完成）

    send_backend = ZmqPushQueue(  # 主进程 → 调度器的消息发送队列
        config.zmq_backend_addr,  # 调度器 backend 端监听的地址
        create=False,  # 不创建（绑定）端点，仅连接已有的
        encoder=BaseBackendMsg.encoder,  # 使用 backend 侧消息编码器序列化
    )

    recv_backend = ZmqPullQueue(  # 调度器 → 主进程的消息接收队列
        config.zmq_detokenizer_addr,  # detokenizer 端地址（主进程在此扮演 detokenizer 接收）
        create=False,  # 仅连接
        decoder=BaseTokenizerMsg.decoder,  # 使用 tokenizer 侧消息解码器反序列化
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")  # 加载与模型匹配的分词器
    prompt = "What's the answer to life, the universe, and everything?"  # 测试 prompt（《银河系漫游指南》经典问题）
    ids = tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)  # 编码为 1D int32 张量（引擎内部的标准格式）
    send_backend.put(
        UserMsg(  # 构造用户请求消息发给调度器
            uid=0,  # 请求 id 为 0
            input_ids=ids,  # 输入 token ids
            sampling_params=SamplingParams(max_tokens=100),  # 最多生成 100 个 token
        )
    )

    while True:  # 循环接收流式生成结果
        msg = recv_backend.get()  # 阻塞取出一条消息
        assert isinstance(msg, DetokenizeMsg)  # 断言消息类型正确（应为逐 token 的反解码消息）
        ids = torch.cat([ids, torch.tensor([msg.next_token], dtype=torch.int32)])  # 把新生成的 token 追加到序列尾部
        if msg.finished:  # 收到结束标志（EOS 或达到 max_tokens）
            break  # 退出接收循环

    logger.info(tokenizer.decode(ids.tolist()))  # 解码完整序列（prompt+输出）并打印
    send_backend.put(ExitMsg())  # 发送退出消息，通知调度器进程结束
