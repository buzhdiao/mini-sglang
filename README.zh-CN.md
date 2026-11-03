> 中文版说明文档，对应原文：[README.md](README.md)（结构与章节保持一致，代码、命令、路径、链接不做翻译）

<p align="center">
<img width="400" src="/assets/logo.png">
</p>

# Mini-SGLang

一个**轻量级却高性能**的大语言模型（LLM）推理框架。

---

Mini-SGLang 是 [SGLang](https://github.com/sgl-project/sglang) 的一个精简实现，旨在揭开现代 LLM 服务（serving）系统的复杂面纱。它以约 **5000 行 Python 代码**的紧凑代码库，既可以作为一个可用的推理引擎，也可以作为研究人员和开发者的透明参考实现。

## ✨ 核心特性

- **高性能**：通过先进的优化技术，达到业界领先的吞吐量与延迟。
- **轻量易读**：代码库干净、模块化且带有完整的类型注解，易于理解和修改。
- **先进优化**：
  - **Radix Cache（基数缓存）**：跨请求复用共享前缀的 KV cache。
  - **Chunked Prefill（分块预填充）**：降低长上下文服务时的峰值内存占用。
  - **Overlap Scheduling（重叠调度）**：用 GPU 计算隐藏 CPU 调度开销。
  - **Tensor Parallelism（张量并行）**：将推理扩展到多张 GPU。
  - **优化算子**：集成 **FlashAttention** 与 **FlashInfer** 以获得极致效率。
  - ...

## 🚀 快速开始

> **⚠️ 平台支持**：Mini-SGLang 目前仅支持 **Linux**（x86_64 和 aarch64）。由于依赖 Linux 专属的 CUDA 内核（`sgl-kernel`、`flashinfer`），不支持 Windows 和 macOS。为了实现跨平台兼容，我们建议 Windows 用户使用 [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) 或 Docker。

### 1. 环境搭建

我们推荐使用 `uv` 进行快速、可靠的安装（注意 `uv` 与 `conda` 并不冲突）。

```bash
# Create a virtual environment (Python 3.10+ recommended)
uv venv --python=3.12
source .venv/bin/activate
```

**前置条件**：Mini-SGLang 依赖 JIT 编译的 CUDA 内核。请确保已安装 **NVIDIA CUDA Toolkit**，且其版本与你的驱动版本匹配。你可以通过 `nvidia-smi` 查看驱动支持的 CUDA 版本。

### 2. 安装

直接从源码安装 Mini-SGLang：

```bash
git clone https://github.com/sgl-project/mini-sglang.git
cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e .
```

<details>
<summary><b>💡 在 Windows（WSL2）上安装</b></summary>

由于 Mini-SGLang 需要 Linux 专属的依赖，Windows 用户应使用 WSL2：

1. **安装 WSL2**（如果尚未安装）：
   ```powershell
   # In PowerShell (as Administrator)
   wsl --install
   ```

2. **在 WSL2 上安装 CUDA**：
   - 参照 [NVIDIA 的 WSL2 CUDA 指南](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
   - 确保你的 Windows GPU 驱动支持 WSL2

3. **在 WSL2 中安装 Mini-SGLang**：
   ```bash
   # Inside WSL2 terminal
   git clone https://github.com/sgl-project/mini-sglang.git
   cd mini-sglang && uv venv --python=3.12 && source .venv/bin/activate
   uv pip install -e .
   ```

4. **从 Windows 访问**：服务器将可以通过 Windows 浏览器和应用在 `http://localhost:8000` 访问。

</details>

<details>
<summary><b>🐳 使用 Docker 运行</b></summary>

**前置条件**：
- [Docker](https://docs.docker.com/get-docker/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

1. **构建 Docker 镜像**：
   ```bash
   docker build -t minisgl .
   ```

2. **启动服务器**：
   ```bash
   docker run --gpus all -p 1919:1919 \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

3. **以交互式 shell 模式运行**：
   ```bash
   docker run -it --gpus all \
       minisgl --model Qwen/Qwen3-0.6B --shell
   ```

4. **使用 Docker Volumes 持久化缓存**（推荐，可加快后续启动速度）：
   ```bash
   docker run --gpus all -p 1919:1919 \
       -v huggingface_cache:/app/.cache/huggingface \
       -v tvm_cache:/app/.cache/tvm-ffi \
       -v flashinfer_cache:/app/.cache/flashinfer \
       minisgl --model Qwen/Qwen3-0.6B --host 0.0.0.0
   ```

</details>

### 3. 在线服务（Online Serving）

一条命令即可启动兼容 OpenAI 的 API 服务器。

```bash
# Deploy Qwen/Qwen3-0.6B on a single GPU
python -m minisgl --model "Qwen/Qwen3-0.6B"

# Deploy meta-llama/Llama-3.1-70B-Instruct on 4 GPUs with Tensor Parallelism, on port 30000
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000
```

服务器启动后，你可以使用 `curl` 等标准工具或任何兼容 OpenAI 的客户端发送请求。

### 4. 交互式 Shell

添加 `--shell` 参数，即可直接在终端中与模型对话。

```bash
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell
```

![shell-example](https://lmsys.org/images/blog/minisgl/shell.png)

你还可以使用 `/reset` 清空对话历史。

## Benchmark（基准测试）

### 离线推理（Offline inference）

详见 [bench.py](./benchmark/offline/bench.py)。设置 `MINISGL_DISABLE_OVERLAP_SCHEDULING=1` 可对 overlap scheduling（重叠调度）进行消融实验。

测试配置：

- 硬件：1xH200 GPU。
- 模型：Qwen3-0.6B、Qwen3-14B
- 总请求数：256 条序列
- 输入长度：在 100-1024 token 之间随机采样
- 输出长度：在 100-1024 token 之间随机采样

![offline](https://lmsys.org/images/blog/minisgl/offline.png)

### 在线推理（Online inference）

详见 [benchmark_qwen.py](./benchmark/online/bench_qwen.py)。

测试配置：

- 硬件：4xH200 GPU，通过 NVLink 互联。
- 模型：Qwen3-32B
- 数据集：[Qwen trace](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon/blob/main/qwen_traceA_blksz_16.jsonl)，重放前 1000 条请求。

启动命令：

```bash
# Mini-SGLang
python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --cache naive

# SGLang
python3 -m sglang.launch_server --model "Qwen/Qwen3-32B" --tp 4 \
    --disable-radix --port 1919 --decode-attention flashinfer
```

> **注意**：如果你在从 HuggingFace 下载模型时遇到网络问题，可尝试使用 `--model-source modelscope` 改从 ModelScope 下载：
> ```bash
> python -m minisgl --model "Qwen/Qwen3-32B" --tp 4 --model-source modelscope
> ```

![online](https://lmsys.org/images/blog/minisgl/online.png)

## 📚 了解更多

- **[详细特性](./docs/features.md)**：探索所有可用的特性与命令行参数。
- **[系统架构](./docs/structures.md)**：深入了解 Mini-SGLang 的设计与数据流。
