> 中文版说明文档，对应原文：[structures.md](structures.md)（结构与章节保持一致，代码、命令、路径、链接不做翻译）

# Mini-SGLang 结构说明

## 系统架构（System Architecture）

Mini-SGLang 被设计为一个分布式系统，以高效处理大语言模型（LLM）推理。它由若干相互独立、协同工作的进程组成。

### 关键组件（Key Components）

- **API Server**：用户入口。它提供兼容 OpenAI 的 API（例如 `/v1/chat/completions`），用于接收 prompt 并返回生成的文本。
- **Tokenizer Worker**：将输入文本转换为模型可以理解的数字（token）。
- **Detokenizer Worker**：将模型生成的数字（token）转换回人类可读的文本。
- **Scheduler Worker**：核心工作进程。在多 GPU 配置下，每张 GPU 对应一个 Scheduler Worker（称为一个 **TP Rank**）。它负责管理该 GPU 上的计算与资源分配。

### 数据流（Data Flow）

各组件之间通过 **ZeroMQ (ZMQ)** 传输控制消息，并通过 **NCCL**（经由 `torch.distributed`）在 GPU 之间交换大规模张量数据。

![Process overview diagram](https://lmsys.org/images/blog/minisgl/design.drawio.png)

**请求生命周期（Request Lifecycle）：**

1. **用户** 向 **API Server** 发送请求。
2. **API Server** 将其转发给 **Tokenizer**。
3. **Tokenizer** 将文本转换为 token，并发送给 **Scheduler (Rank 0)**。
4. **Scheduler (Rank 0)** 将请求广播给所有其他 Scheduler（如果使用多张 GPU）。
5. **所有 Scheduler** 对请求进行调度，并触发各自的本地 **Engine** 计算下一个 token。
6. **Scheduler (Rank 0)** 收集输出 token，并发送给 **Detokenizer**。
7. **Detokenizer** 将 token 转换为文本，并回传给 **API Server**。
8. **API Server** 将结果以流式方式返回给 **用户**。

## 代码组织（`minisgl` 包）

源代码位于 `python/minisgl`。以下是为开发者整理的模块说明：

- `minisgl.core`：提供核心数据类 `Req` 和 `Batch`，表示请求的状态；提供类 `Context`，保存推理上下文的全局状态；类 `SamplingParams` 保存用户提供的采样参数。
- `minisgl.distributed`：提供张量并行中 all-reduce 和 all-gather 的接口，以及数据类 `DistributedInfo`，用于保存 TP worker 的 TP 信息。
- `minisgl.layers`：实现构建 LLM 的基础组件，均支持 TP，包括 linear（线性层）、layernorm、embedding、RoPE 等。它们共享 `minisgl.layers.base` 中定义的公共基类。
- `minisgl.models`：实现 LLM 模型，包括 Llama 和 Qwen3。还定义了从 huggingface 加载权重以及对权重进行分片（sharding）的工具函数。
- `minisgl.attention`：提供 attention backend 的接口，并实现了 `flashattention` 和 `flashinfer` 后端。它们由 `AttentionLayer` 调用，并使用存储在 `Context` 中的元数据。
- `minisgl.kvcache`：提供 KVCache pool（池）与 KVCache manager（管理器）的接口，并实现了 `MHAKVCache`、`NaiveCacheManager` 和 `RadixCacheManager`。
- `minisgl.utils`：提供一组工具集合，包括 logger 配置以及 zmq 的封装。
- `minisgl.engine`：实现 `Engine` 类，即单个进程上的一个 TP worker。它管理模型、context、KVCache、attention backend 以及 CUDA graph 重放。
- `minisgl.message`：定义在 api_server、tokenizer、detokenizer 与 scheduler 之间（经由 zmq）交换的消息。所有消息类型均支持自动序列化与反序列化。
- `minisgl.scheduler`：实现 `Scheduler` 类，运行在每个 TP worker 进程上，管理对应的 `Engine`。rank 0 的 scheduler 接收来自 tokenizer 的消息，与其他 TP worker 上的 scheduler 通信，并向 detokenizer 发送消息。
- `minisgl.server`：定义命令行参数，以及启动 Mini-SGLang 所有子进程的 `launch_server`。还在 `minisgl.server.api_server` 中实现了一个作为前端的 FastAPI 服务器，提供诸如 `/v1/chat/completions` 的端点。
- `minisgl.tokenizer`：实现 `tokenize_worker` 函数，处理 tokenize（编码）与 detokenize（解码）请求。
- `minisgl.llm`：提供 `LLM` 类，作为与 Mini-SGLang 系统便捷交互的 Python 接口。
- `minisgl.kernel`：实现自定义 CUDA 内核，通过 `tvm-ffi` 提供 Python 绑定与 JIT 接口。
- `minisgl.benchmark`：基准测试工具。
