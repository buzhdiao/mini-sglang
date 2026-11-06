# functools.cache：配置对象按路径缓存（多处读取不重复解析）
import functools
# json：读取独立存放的 chat_template 文件
import json
# os：本地目录判断
import os
# 类型注解工具
from typing import Any

# HF Hub 下载工具
from huggingface_hub import hf_hub_download, snapshot_download
# tqdm：进度条基类（用于派生禁用版）
from tqdm.asyncio import tqdm
# transformers：自动配置/分词器与基类
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

# 下载权重时不弹进度条（多进程日志场景下进度条会刷屏）
class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        # 移除调用方可能传入的 name 参数（基类不认）
        kwargs.pop("name", None)
        # 强制禁用进度条渲染
        kwargs["disable"] = True
        # 交给基类初始化
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    # 从本地目录或 HF repo id 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    # 部分 Mistral 模型把聊天模板存在单独的 JSON 文件里，需要补加载
    if not getattr(tokenizer, "chat_template", None):
        try:
            # 从 Hub 下载 chat_template.json（本地缓存命中则直接返回）
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                # 读出模板并挂到 tokenizer 上
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            # 没有该文件或解析失败：静默放弃（无模板也能做普通分词）
            pass
    # 返回就绪的 tokenizer
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    # 真正加载 HF 配置（缓存保证同路径只解析一次）
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    # 取缓存的配置对象
    config = _load_hf_config(model_path)
    # 用 dict 重建一份全新副本：防止调用方修改缓存里的共享对象
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        # 已是本地目录：无需下载，直接返回
        return model_path
    try:
        # 按 repo id 下载权重快照（只拉 safetensors 格式，静默进度条）
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        # 既不是本地目录也下载失败：报出带原因的明确错误
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )