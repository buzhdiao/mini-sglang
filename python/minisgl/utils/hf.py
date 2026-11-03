import functools  # 提供 cache 装饰器，缓存 HF 配置的加载结果
import json  # 解析 Mistral 风格的 chat_template.json 文件
import os  # 判断模型路径是否为本地目录
from typing import Any  # 类型注解：任意类型

from huggingface_hub import hf_hub_download, snapshot_download  # HF Hub 下载接口：单文件下载 / 整仓快照下载
from tqdm.asyncio import tqdm  # 进度条基类，这里仅用于继承并禁用
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase  # HF 配置与分词器的加载入口及基类

class DisabledTqdm(tqdm):
    """静默进度条：继承 tqdm 但强制禁用输出，避免下载权重时刷屏干扰日志"""

    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)  # hf_hub 会传入 name 参数，tqdm 不认，先移除避免报错
        kwargs["disable"] = True  # disable=True 让进度条完全静默
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    """加载分词器；对部分 Mistral 模型额外从 chat_template.json 补充聊天模板"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)  # 按本地路径或 Hub ID 加载对应分词器
    # Some Mistral models store chat_template in a separate JSON file
    # 部分 Mistral 模型把聊天模板存在独立的 JSON 文件中，分词器主配置里没有
    if not getattr(tokenizer, "chat_template", None):  # 分词器本身不带 chat_template 时才尝试补充
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")  # 从 Hub 下载模板文件（有本地缓存则直接命中）
            with open(path, "r", encoding="utf-8") as f:  # 读取下载/缓存到的 JSON 文件
                tokenizer.chat_template = json.load(f)["chat_template"]  # 取出模板字段挂回分词器对象
        except Exception:  # 文件不存在或解析失败都不致命——只是没有聊天模板可用
            pass  # 静默忽略，保持 tokenizer 原样返回
    return tokenizer


@functools.cache  # 缓存：同一 model_path 的配置只解析一次（配置解析涉及磁盘 IO，较昂贵）
def _load_hf_config(model_path: str) -> Any:
    """内部函数：加载并缓存 HF 模型配置对象"""
    return AutoConfig.from_pretrained(model_path)  # 按模型路径/ID 自动识别并加载配置类


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    """对外接口：返回配置的"重建副本"，避免调用方修改污染 functools.cache 缓存对象"""
    config = _load_hf_config(model_path)  # 命中缓存的配置对象（内部共享）
    return type(config)(**config.to_dict())  # 用原配置类从字典重建一个新实例，实现隔离拷贝


def download_hf_weight(model_path: str) -> str:
    """解析并下载模型权重，返回本地权重目录路径"""
    if os.path.isdir(model_path):  # 已是本地目录（如离线/自训模型），无需下载
        return model_path  # 原样返回该路径
    try:
        return snapshot_download(  # 按 Hub 快照方式把整个仓库下载到本地缓存目录
            model_path,
            allow_patterns=["*.safetensors"],  # 只下载 safetensors 权重（跳过 .bin 等冗余格式，省流量省磁盘）
            tqdm_class=DisabledTqdm,  # 用静默进度条，避免多进程场景日志混乱
        )
    except Exception as e:
        raise ValueError(  # 转成更友好的错误：既不是本地目录，也不是合法的 Hub 模型 ID
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )
