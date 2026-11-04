# Mini-SGLang 学习计划 · 题目与知识点整理

> 来源：[lynnyulinlin-debug/cpp-llm-engine](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues) 的 issues（截至 2026-11-04，共 19 个，其中 15 个为学习计划类）
> 整理规则：
> - **保留**：各期学习计划 issue 中的核心目标、知识点、源码阅读材料、思考题与参考链接，题目原文保留，并附加「考察点 / 解题思路」注解。
> - **排除**：明确属于「上机 / 实验 / 需要运行代码 / 需要环境」的内容——5 个打卡类 issue（#7、#8、#9、#12、#19）整体排除；各计划 issue 内的「动手任务」「打卡模板」「打卡截图要求」「GPU 环境自检」环节剔除。排除明细见附录 B。

---

## 目录

- [概述：Mini-SGLang 是什么](#概述mini-sglang-是什么)
- [第 0 部分 前置任务：不同基础学习者的准备题（issue #6）](#第-0-部分-前置任务不同基础学习者的准备题issue-6)
- [第一期学习计划（Task 0-6，issues #1-5、#10、#11）](#第一期学习计划task-0-6)
  - [Task 0 推理基础与 SGLang 定位](#第一期-task-0-推理基础与-sglang-定位)
  - [Task 1 模型层与核心数据结构](#第一期-task-1-模型层与核心数据结构)
  - [Task 2 KV Cache 与 Paged Attention](#第一期-task-2-kv-cache-与-paged-attention)
  - [Task 3 调度器与 Continuous Batching](#第一期-task-3-调度器与-continuous-batching)
  - [Task 4 Radix Cache 与前缀复用](#第一期-task-4-radix-cache-与前缀复用)
  - [Task 5 Tensor Parallelism 与服务层](#第一期-task-5-tensor-parallelism-与服务层)
  - [Task 6 整合与扩展](#第一期-task-6-整合与扩展)
- [第二期学习计划（Task 0-4，issues #14-18）：Mini vs SGLang 对照](#第二期学习计划task-0-4mini-vs-sglang-对照)
  - [Task 0 推理基础与定位](#第二期-task-0-推理基础与定位)
  - [Task 1 KV Cache 与 Paged Attention](#第二期-task-1-kv-cache-与-paged-attention)
  - [Task 2 调度器与 Continuous Batching](#第二期-task-2-调度器与-continuous-batching)
  - [Task 3 RadixAttention 与前缀复用](#第二期-task-3-radixattention-与前缀复用)
  - [Task 4 Overlap Scheduling 与 CPU-GPU 流水线](#第二期-task-4-overlap-scheduling-与-cpu-gpu-流水线)
- [附录 A 第一期任务总索引（issue #13）](#附录-a-第一期任务总索引issue-13)
- [附录 B 排除清单](#附录-b-排除清单)

---

## 概述：Mini-SGLang 是什么

（引自各期 issue 的共同导语，原文）

> Mini-SGLang 是 SGLang 的轻量版实现，约 5,000 行 Python 代码，保留了 Radix Cache、Chunked Prefill、Overlap Scheduling、Tensor Parallelism 等核心特性，适合理解现代 LLM 服务系统的工程实现。仅支持 Linux（x86_64/aarch64），Windows 需使用 WSL2。

第二期补充：

> 你可以先在 Python 层面把调度、缓存、批处理的核心逻辑吃透，再无缝对照 SGLang 的生产级实现（如 UnifiedRadixTree、HiCache 多级缓存）。

---

## 第 0 部分 前置任务：不同基础学习者的准备题（issue #6）

> 原文：[issue #6](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/6)

### A 类人群：完全没用过推理框架 / 原理不熟

目标：建立推理框架全局观。

| 题号 | 题目原文 |
|---|---|
| A1 | 模型、Tokenizer、推理框架、GPU、API 服务之间的关系。 |
| A2 | 说明推理两阶段 Prefill 和 Decode，各自是计算密集还是带宽密集。 |
| A3 | 说明两个指标 TTFT 和 TPOT，分别对应哪个阶段。 |

（A4 为 GPU 环境自检，属环境类，已剔除，见附录 B）

**考察点**：推理系统分层架构；两阶段推理的算力/带宽特征；服务性能指标定义。

**解题思路**：
- A1：一次 HTTP 请求的典型链路是 API Server（接收/解析请求）→ Tokenizer（文本 ↔ token id）→ 推理框架（调度、批处理、KV Cache 管理）→ GPU（执行模型前向计算）→ Detokenizer/流式回传。模型是"给 token 计算下一个 token"的核心，框架是模型执行的容器与调度器。
- A2：Prefill 一次处理整个 prompt，是大矩阵乘，**计算密集**（compute-bound）；Decode 每步只生成 1 个 token 却要读取全部权重与 KV Cache，**带宽密集**（memory-bound）。
- A3：TTFT（Time To First Token）对应 **Prefill** 阶段；TPOT（Time Per Output Token）对应 **Decode** 阶段。

### B 类人群：用过 vLLM，但没用过 SGLang

目标：能对比 vLLM 与 SGLang。

| 题号 | 题目原文 |
|---|---|
| B1 | 说明 SGLang 的 RadixAttention 与 vLLM 的 PagedAttention 有什么异同。 |
| B2 | 从调度、KV Cache、Attention 后端、API、模型支持、硬件支持、量化、部署复杂度做一张对比表。 |
| B3 | 调研二者对 Qwen / DeepSeek 的支持版本和环境要求。 |

**考察点**：两大主流推理框架的核心机制差异与选型能力。

**解题思路**：
- B1：**相同点**——都做 KV Cache 的细粒度管理以减少碎片、支持连续批处理。**差异**——PagedAttention 解决的是"如何分页存储 KV Cache"（物理块表映射，类似操作系统虚存）；RadixAttention 在此之上解决"如何跨请求复用 KV Cache"（用 Radix Tree 按前缀组织缓存，命中即免重算）。二者正交互补，SGLang 内部同样使用分页式的 KV 存储。
- B2：对比维度提示——调度：SGLang 有前缀感知的 LPM 调度 vs vLLM 的预算式调度；KV Cache：前缀复用 vs 纯分页；其余维度（API 兼容 OpenAI 协议、量化方案、硬件支持等）可查阅两仓库 README/文档完成表格。
- B3：以官方 README 与模型支持列表为准，关注最小版本要求与推荐配置。

### C 类人群：用过 SGLang，想深入 mini-sglang

目标：定位 mini-sglang 的简化点与核心机制。

| 题号 | 题目原文 |
|---|---|
| C1 | 阅读 mini-sglang README 和目录结构。 |
| C2 | 列出它相比 SGLang 简化/省略的 3 个生产级能力。 |
| C3 | 列出它重点实现的 3 个核心机制。 |

**解题思路**：参考本仓库 `docs/structures.zh-CN.md`、`docs/features.zh-CN.md`。C2 可从以下角度取三：多级缓存（HiCache L1/L2/L3）、分布式服务架构（多 tokenizer/scheduler 进程部署）、推测解码/结构化输出等高级特性。C3 通常答：Radix Cache、Chunked Prefill、Overlap Scheduling（或 TP）。

### 公共问题（ABC 类通用，原文）

> 1. SGLang 与 mini-sglang 是什么关系？（提示：从代码规模、依赖、分布式、调度、Attention 后端、KV Cache、HTTP 服务、教学目的几个角度回答。）
> 2. mini-sglang 的运行环境和最小入门模型是什么？

**解题思路**：
- 1：mini-sglang 是 SGLang 的教学精简版——单进程简化架构、代码量约 5k 行、保留四大核心机制，用于学习原理；SGLang 是生产级引擎，功能全集且持续演进。
- 2：Linux（x86_64/aarch64）+ Python 3.10+ + NVIDIA GPU（CUDA Toolkit 用于 JIT 编译 kernel）；最小入门模型 **Qwen3-0.6B**，单卡 ≥8GB 显存即可。
（公共问题 3 为个人 GPU 环境自检，属环境类，已剔除，见附录 B。）

---

## 第一期学习计划（Task 0-6）

> 共 7 讲，总索引见 [issue #13](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/13)（附录 A 有链接表）。

<a id="第一期-task-0-推理基础与-sglang-定位"></a>
### 第一期 Task 0 推理基础与 SGLang 定位

> 原文：[issue #1](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/1)
> **核心目标**（原文）：理解 LLM 推理的本质问题，建立系统视角。

**知识点与学习材料**（原文）：
- Quick Start：https://github.com/sgl-project/mini-sglang
- 阅读 Mini-SGLang 的 Code Structure 文档，理解 core（Req/Batch/Context）、engine、layers、models、kvcache 等模块的职责划分：https://mintlify.wiki/sgl-project/mini-sglang/api/code-structure

**思考题**（原打卡项的知识性内容）：
1. 用自己的话解释 KV Cache、Prefill/Decode、内存墙（memory wall）。
2. SGLang 与 vLLM 的定位对比。

**考察点**：推理系统全局观；核心概念的准确理解。

**解题思路**：
- KV Cache：把每层注意力中已算过的 K/V 缓存下来，避免每生成一个 token 就对整段序列重算注意力；代价是显存随序列长度线性增长（这正是 Task 2 的主题）。
- Prefill/Decode：见上文 A2。
- 内存墙：GPU 算力增长远快于显存带宽增长，Decode 阶段每 token 都要完整读权重，算力空转——这是 Continuous Batching、量化、多级缓存等优化的共同动机。

<a id="第一期-task-1-模型层与核心数据结构"></a>
### 第一期 Task 1 模型层与核心数据结构

> 原文：[issue #2](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/2)
> **核心目标**（原文）：理解 Mini-SGLang 如何加载和表示一个 LLM。

**知识点与源码阅读**（原文）：
- `minisgl.models`：BaseLLMModel 基类、ModelConfig 配置、Llama/Qwen2/Qwen3 的具体实现
- `minisgl.layers`：VocabParallelEmbedding、LinearQKVMerged、RMSNorm、RoPE、AttentionLayer
- `minisgl.core`：Req（请求状态）、Batch（批处理分组）、Context（全局推理上下文）

**思考题**：
1. 基类与配置如何配合使用？请求（Req）与批（Batch）如何构造？
2. 画出模型加载流程图与核心数据结构关系图。

**考察点**：模型抽象（配置驱动 + 注册机制）；请求生命周期中三大核心数据结构的职责边界。

**解题思路**：ModelConfig 承载超参（层数/头数/hidden_size/dtype 等），BaseLLMModel 定义 `load_weights`/`forward` 协议，具体模型（Llama/Qwen2/Qwen3/Qwen3MoE）经 register 注册、按 config.architectures 分派；Req 持有 prompt/输出 token 序列与状态机（waiting→running→finished），Batch 是同一调度步内多个 Req 的张量视图集合，Context 串起 model/tokenizer/kvcache 等全局单例。

<a id="第一期-task-2-kv-cache-与-paged-attention"></a>
### 第一期 Task 2 KV Cache 与 Paged Attention

> 原文：[issue #3](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/3)
> **核心目标**（原文）：理解推理引擎最核心的内存管理机制。

**知识点与源码阅读**（原文）：
- KV Cache 的内存占用公式、Paged KV Cache 的块分配原理
- 阅读 `minisgl.kvcache` 模块，理解 BlockAllocator 的分配/释放逻辑
- 理解 FlashAttentionBackend 和 FlashInferBackend 的接口抽象

**思考题**：
1. 写出 KV Cache 内存占用公式；计算不同序列长度下 Llama-3-8B 的 KV Cache 大小。
2. 画出 Paged KV Cache 内存布局示意图。
3. BlockAllocator 如何分配/释放？FA 与 FI 后端的接口抽象有何异同？

**考察点**：显存定量估算；虚拟化/分页思想在 KV Cache 上的落地；后端抽象设计。

**解题思路**：
- 公式：`KV(bytes) = 2 (K和V) × num_layers × kv_heads × head_dim × dtype_size × seq_len`（GQA 用 kv_heads，MQA 为 1）。Llama-3-8B：32 层 × 8 kv_heads × 128 head_dim × 2 bytes(bf16) = 128 KB/token；8K 序列约 1 GB/请求。
- 分页原理：把每序列的 KV 按固定 block_size（如 16 token）分块，逻辑连续、物理离散，用 block table 映射；请求结束时按块释放，消除外部碎片。
- 后端抽象：两者都暴露 `init_forward_metadata / forward_extend / forward_decode` 类接口；FA 基于自研 kernel，FI 是算子库（还提供 plan 一步），Mini 用统一接口屏蔽差异。

<a id="第一期-task-3-调度器与-continuous-batching"></a>
### 第一期 Task 3 调度器与 Continuous Batching

> 原文：[issue #4](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/4)
> **核心目标**（原文）：理解请求如何在引擎中被调度和执行。

**知识点与源码阅读**（原文）：
- 阅读 `minisgl.engine` 中的 Engine 类，理解 Prefill 队列和 Decode 队列的调度策略
- Continuous Batching：理解"请求动态加入/退出批次"的机制，以及它相比 Static Batching 的优势
- Chunked Prefill：理解 Chunked Prefill 如何控制长上下文场景下的峰值内存占用

**思考题**：
1. 画出调度流程时序图。
2. Continuous Batching 与 Static Batching 的对比分析。

**考察点**：批处理调度的时空利用率分析；长短请求混合下的公平性与吞吐。

**解题思路**：
- Static Batching：一批请求必须全部完成后才能组下一批，短请求被长请求"拖住"，GPU 空转严重；Continuous Batching 在每个 decode 步之间动态换入换出请求，完成一个立刻补位一个，吞吐可提升数倍到数十倍（vLLM 论文的核心结论）。
- Chunked Prefill：把超长 prompt 切成 ≤max_prefill_tokens 的多个 chunk 分多步处理，把 Prefill 的激活峰值内存从 O(整段序列) 压到 O(chunk)，同时避免一条长 prompt 独占整个调度步、阻塞所有 Decode 请求。

<a id="第一期-task-4-radix-cache-与前缀复用"></a>
### 第一期 Task 4 Radix Cache 与前缀复用

> 原文：[issue #5](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/5)
> **核心目标**（原文）：理解 SGLang 最具特色的优化——RadixAttention。

**知识点与源码阅读**（原文）：
- （1）Radix Tree 数据结构 （2）前缀共享的原理 （3）为什么多轮对话和 Few-shot 场景受益最大
- Overlap Scheduling：理解 Overlap Scheduling 如何隐藏 CPU 调度开销，将其与 GPU 计算重叠
- 阅读 minisgl 中 Radix Cache 的实现，理解 KV Cache 如何在前缀级别复用

**思考题**：
1. 说明 Radix Tree 结构、前缀共享原理、以及多轮对话/Few-shot 受益最大的原因。
2. 设计实验：同一前缀的多个请求 vs 不同前缀的多个请求，对比吞吐量差异。

**考察点**：缓存复用的算法结构与收益边界；CPU-GPU 流水线思想。

**解题思路**：
- Radix Tree 以 token 序列为键，公共前缀共享一条路径上的物理 KV 块（引用计数管理生命周期）；命中前缀的请求只需计算差异部分。
- 多轮对话的新请求 = 旧前缀 + 少量新 token，Few-shot 的公共 system prompt/示例被所有请求共享——两者的前缀重复率最高，因此收益最大。逐 token 采样生成的随机序列则几乎无法命中。
- Overlap Scheduling：把"下一批的 CPU 调度（组批、分配块、构建元数据）"与"当前批的 GPU 前向"重叠执行，CPU 调度延迟被 GPU 计算时间吸收，消除每步的空闲气泡。

<a id="第一期-task-5-tensor-parallelism-与服务层"></a>
### 第一期 Task 5 Tensor Parallelism 与服务层

> 原文：[issue #10](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/10)
> **核心目标**（原文）：理解多 GPU 扩展和服务部署。

**知识点与源码阅读**（原文）：
- 分布式模块：阅读 `minisgl.distributed`：(1) DistributedInfo（TP worker 信息）、(2) DistributedCommunicator（All-Reduce/All-Gather）、(3) NCCL 集成
- 张量并行：理解 VocabParallelEmbedding、LinearRowParallel、LinearColParallelMerged 的并行策略
- 服务架构：理解 API Server → Tokenizer → Scheduler → Engine → Detokenizer 的完整链路

**思考题**：
1. 说明三个分布式组件的职责；说明三种并行切分策略。
2. 画出张量并行通信模式图与服务请求全链路时序图。

**考察点**：TP 的数学原理（矩阵分块与通信点的对应）；多进程服务架构的职责拆分。

**解题思路**：
- 列并行（ColumnParallel，如 QKV/MLP up 投影）：权重按**输出维**切分，各卡算得输出的一个分片，无需通信；行并行（RowParallel，如 MLP down/o_proj）：权重按**输入维**切分，各卡算得部分和，需要一次 **All-Reduce** 求和。VocabParallelEmbedding 按词表维切分 embedding，未命中本卡分片的 token 置零，最后 All-Reduce。
- Transformer 层的标准 TP 布局：attention/MLP 的前一个投影用列并行、后一个投影用行并行，使每层只需 2 次 All-Reduce（前向）。
- 服务链路按进程拆分（ZeroMQ 消息传递）：API Server 解析 HTTP → Tokenizer worker 转token → Scheduler 组批调度 → Engine 在 GPU 执行 → Detokenizer 解码 → 流式回传，CPU 密集环节与 GPU 计算解耦。

<a id="第一期-task-6-整合与扩展"></a>
### 第一期 Task 6 整合与扩展

> 原文：[issue #11](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/11)
> **核心目标**（原文）：将 Task 0-6 所学串联，完成一个可展示的学习成果。

**知识点与学习材料**（原文）：
- 源码对比：将 Mini-SGLang 与你了解的 vLLM 或真实 SGLang 做架构对比
- 扩展阅读：了解后续可深入的方向（量化、分层缓存、PD 分离等）

**思考题**：
1. 撰写《Mini-SGLang 源码学习报告》：架构总览 + 核心模块分析 + 与真实 SGLang 的对比。
2. 分析几个扩展方向：量化、分层缓存、PD 分离。

**解题思路**（扩展方向要点）：
- 量化：权重/激活降至 INT8/FP8/INT4，缓解 Decode 的权重带宽瓶颈与显存容量压力；推理侧还需保留 FP8 KV Cache 等方案。
- 分层缓存：SGLang HiCache 把 KV Cache 分为 L1（显存）/L2（本机内存）/L3（远端/磁盘），前缀命中可跨层找回，突破显存容量限制。
- PD 分离（Prefill/Decode disaggregation）：Prefill 是计算密集、Decode 是带宽密集，分别部署到不同实例（如 Mooncake/DeepSeek 3FS 架构），中间传输 KV Cache，消除两类负载互相干扰。

---

## 第二期学习计划（Task 0-4，Mini vs SGLang 对照）

> 第二期的主线是「Mini 的简化实现 vs SGLang 的生产级实现」逐点对照。第二期 Task 编号与第一期不同（第二期 Task 1 即 KV Cache 主题）。

<a id="第二期-task-0-推理基础与定位"></a>
### 第二期 Task 0 推理基础与定位

> 原文：[issue #14](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/14)
> **核心目标**（原文）：建立推理引擎的问题意识，理解 Mini-SGLang 与 SGLang 的定位关系。

**知识点与学习材料**（原文）：
- Mini-SGLang 的 Code Structure 文档：https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md 、 https://github.com/sgl-project/mini-sglang/blob/main/docs/features.md
- Mini-SGLang 的 Code Structure 文档 vs SGLang 的架构概览
- 查阅高频概念：KV Cache、Prefill/Decode、Continuous Batching、RadixAttention

**思考题**：
1. 说明 mini-sglang 的模块职责划分。
2. 画一张"从 Mini-SGLang 到 SGLang"的架构演进图，标注哪些模块在 Mini 中被简化、在完整版中被扩展。
3. 解释高频概念（KV Cache、Prefill/Decode、Continuous Batching、RadixAttention）的原理及其解决的问题。

**解题思路**：演进图可按层标注——调度：单进程 Scheduler → 多进程调度器集群 + 预算驱动准入；KV Cache：单级 Radix Cache → UnifiedRadixTree + HiCache 多级缓存；Attention：FA/FI 双后端 → 全后端矩阵（FlashAttention/FlashInfer/Triton/Flex/FA3…）；服务：单 API Server → 多 tokenizer/多 detokenizer + 负载均衡。四个高频概念的原理见第一期各 Task 的解题思路。

<a id="第二期-task-1-kv-cache-与-paged-attention"></a>
### 第二期 Task 1 KV Cache 与 Paged Attention

> 原文：[issue #15](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/15)
> **核心目标**（原文）：理解推理引擎最核心的内存管理机制，从 Mini 的简化实现过渡到 SGLang 的完整设计。

**知识点与源码阅读**（原文）：
- Mini 侧：阅读 `minisgl.kvcache` 的 BlockAllocator，理解分页 KV Cache 的分配/释放逻辑。
- SGLang 侧：对照 SGLang 的 KV Pool + Radix Tree + ReqToTokenPool 三层结构，理解物理存储与逻辑索引的分离。

**思考题**：
1. Mini 的 Block 分配与 SGLang 的 HiCache 分层设计（L1/L2/L3）之间，多了哪些工程考量？写 KV Cache 内存布局对比笔记。

**考察点**：从"够用的分页"到"容量可扩展的分层体系"之间的工程差距。

**解题思路**：HiCache 的三层各自解决不同约束——L1（GPU 显存）保速率、L2（Host 内存）扩容量、L3（远端存储/多实例共享）扩规模与跨实例复用；由此引入的问题包括：层间传输带宽管理、写回/预取策略、一致性（页失效与更新）、TP 一致性哈希等，这些都是单级 BlockAllocator 不需要考虑的。

<a id="第二期-task-2-调度器与-continuous-batching"></a>
### 第二期 Task 2 调度器与 Continuous Batching

> 原文：[issue #16](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/16)
> **核心目标**（原文）：理解请求如何被调度、批处理如何动态形成。

**知识点与源码阅读**（原文）：
- Mini 侧：阅读 Mini 中 Radix Cache 的实现，理解前缀共享的基本原理。
- SGLang 侧：对照 UnifiedRadixTree（统一 FULL/MAMBA/SWA 索引）和 HiRadixTree（分层缓存），理解 Radix Tree 如何从"单机前缀复用"演化为"多级缓存协调器"。

**思考题**：
1. Mini 的简单队列与 SGLang 的预算驱动调度之间，多了哪些对真实负载的适应性设计？画调度流程时序图（Mini vs SGLang）。
2. 解释 Continuous Batching 机制及相对 Static Batching 的优势；Chunked Prefill 如何控制长上下文的峰值内存。

**解题思路**：SGLang 的预算驱动调度以 PrefillAdder 为核心，包含多种准入预算（rem total tokens、rem input tokens、rem chunk tokens、rem input bseqlens、rem logits tokens 等），在"吞吐最大化"与"内存不爆、公平不饿死"之间动态权衡，并支持 in-batch prefix caching、优先级/取消等真实负载需求；Mini 只维护简单的 waiting 队列按序组批。

<a id="第二期-task-3-radixattention-与前缀复用"></a>
### 第二期 Task 3 RadixAttention 与前缀复用

> 原文：[issue #17](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/17)
> **核心目标**（原文）：深入 SGLang 最具标志性的优化——跨请求 KV Cache 复用。

**知识点与源码阅读**（原文）：
- Mini 侧：阅读 Engine 的 Prefill/Decode 队列调度逻辑，理解 Continuous Batching 的基本机制。
- 阅读 minisgl 中 Radix Cache 的实现，理解 KV Cache 如何在前缀级别复用。
- SGLang 侧：对照 SGLang 调度器的 Prefill > Decode 优先级决策、PrefillAdder 五种准入预算、Chunked Prefill 的截断逻辑。

**思考题**：
1. 说明（1）Radix Tree 数据结构（2）前缀共享的原理（3）为什么多轮对话和 Few-shot 场景受益最大。
2. RadixAttention 的缓存命中率受哪些因素影响？SGLang 的 LPM（最长前缀匹配）调度策略如何与之配合？

**考察点**：缓存命中率的影响因素分析与调度协同。

**解题思路**：命中率影响因素——请求流的前缀分布（是否共享 system prompt/多轮历史）、缓存容量与淘汰策略（LRU + 锁定引用）、逐 token 采样带来的分支发散、负载淘汰压力；LPM 调度在挑选下一批 prefill 请求时优先选择与当前缓存匹配最长前缀的请求，把"缓存里有什么"反馈到"让谁先跑"，进一步提高命中。

<a id="第二期-task-4-overlap-scheduling-与-cpu-gpu-流水线"></a>
### 第二期 Task 4 Overlap Scheduling 与 CPU-GPU 流水线

> 原文：[issue #18](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/18)
> **核心目标**（原文）：理解如何隐藏 CPU 调度开销，让 GPU 持续忙碌。

**知识点与源码阅读**（原文）：
- Mini 侧：阅读 Overlap Scheduling 的实现，理解"CPU 批组织与 GPU 计算重叠"的基本思路。
- SGLang 侧：对照 SGLang 的 result_queue + FutureMap relay 双流水线机制，理解其如何打破串行依赖。

**思考题**：
1. Mini 的简单重叠与 SGLang 的异步流水线之间，对延迟和吞吐的改善差异在哪里？
2. 画 CPU-GPU 重叠调度流程图 + 关键数据依赖分析。

**解题思路**：朴素重叠是两级流水（调度第 N+1 步的同时 GPU 算第 N 步），但采样结果回传/反序列化仍会插入等待；SGLang 用 result_queue + FutureMap 把"结果异步 relay、future 占位、下轮调度不阻塞"组合成更深的流水线，调度线程几乎从不等待 GPU，瓶颈从调度彻底转移回 GPU 本身。关键依赖：下一批输入只依赖上一批的 token id（可先给占位 future），不依赖完整的 detokenize/传输结果。

---

## 附录 A 第一期任务总索引（issue #13）

原文（[issue #13](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/13)）为第一期 7 讲的链接总索引：

| 讲次 | 主题 | 原文链接 |
|---|---|---|
| Task 0 | 推理基础与 SGLang 定位 | https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/1 |
| Task 1 | 模型层与核心数据结构 | https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/2 |
| Task 2 | KV Cache 与 Paged Attention | https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/3 |
| Task 3 | 调度器与 Continuous Batching | https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/4 |
| Task 4 | Radix Cache 与前缀复用 | https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/5 |
| Task 5 | Tensor Parallelism 与服务层 | https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/10 |
| Task 6 | 整合与扩展 | https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/11 |

第二期（issues #14-18）同样依次对应 Task 0-4（主题见上文目录）。

## 附录 B 排除清单

**整体排除的 issue（上机/打卡类，共 5 个）：**

| Issue | 标题 | 排除原因 |
|---|---|---|
| [#7](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/7) | mini_sglang 前置任务完成 | 上机打卡报告（环境信息、nvidia-smi、完成情况汇报） |
| [#8](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/8) | Task 0 打卡 | 打卡类 |
| [#9](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/9) | 打卡 | 打卡类 |
| [#12](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/12) | 打卡 | 打卡类 |
| [#19](https://github.com/lynnyulinlin-debug/cpp-llm-engine/issues/19) | Task0 打卡 | 打卡类（含运行截图与环境信息） |

**各计划 issue 内剔除的环节：**
- 所有 issue 的「打卡模板」「打卡截图要求」小节（微信群截图规范等）；
- 各期的「动手任务」小节（安装环境、启动服务、运行脚本、截图取证等）；
- issue #1/2/3 等打卡项中"运行截图"类子项（仅保留其中的知识性描述）；
- issue #6 中 A4/B4/C4（GPU 环境自检/环境准备）与公共问题 3（个人环境是否满足）。
