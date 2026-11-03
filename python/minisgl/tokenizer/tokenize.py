from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解使用前向引用

from typing import List  # 导入类型注解 List，用于声明消息列表与结果列表类型

import torch  # PyTorch，用于构造 int32 的输入 token 张量
from minisgl.message import TokenizeMsg  # tokenize 请求消息：携带 uid、原始文本/对话列表与采样参数
from transformers import PreTrainedTokenizerBase  # HuggingFace 分词器基类，统一不同模型的 encode 接口


class TokenizeManager:
    """tokenize 管理器：把用户文本（或对话消息列表）编码成模型可用的输入 id 张量"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer  # 保存分词器实例，供后续 encode 调用复用（避免重复加载）

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        """批量处理 tokenize 请求，返回与 msgs 一一对应的输入 id 张量列表"""
        results: List[torch.Tensor] = []  # 结果容器，保持与 msgs 相同的顺序
        # TODO: batch tokenization  # TODO：当前逐条编码，可改为 tokenizer(pad_to_multiple) 批量编码提升吞吐
        for msg in msgs:  # 逐条处理每个 tokenize 请求
            if isinstance(msg.text, list):  # text 是消息列表时，说明是 chat 请求，需要套用对话模板
                prompt = self.tokenizer.apply_chat_template(  # 将对话消息列表渲染成最终 prompt 字符串
                    msg.text,  # 原始的 [{role, content}, ...] 消息列表
                    tokenize=False,  # 只返回模板渲染后的文本，不在模板阶段分词（后面统一 encode）
                    add_generation_prompt=True,  # 末尾追加 assistant 起始标记，引导模型开始生成回复
                )
                assert isinstance(prompt, str)  # 确保模板返回字符串（某些模板/参数组合可能返回 token 列表）
            else:
                prompt = msg.text  # text 是纯字符串时（如 /generate 接口），直接作为 prompt 使用
            input_ids: torch.Tensor = (  # type: ignore  # 显式标注类型；transformers 返回类型标注不精确故忽略检查
                self.tokenizer.encode(prompt, return_tensors="pt")  # 编码 prompt 并直接返回 PyTorch 张量（形状 [1, L]）
            )
            results.append(input_ids.view(-1).to(torch.int32))  # 展平成 [L] 并转 int32（引擎内部 KV/词表索引按 32 位处理，省显存）
        return results  # 返回所有输入 id 张量，交给上层打包成 BatchBackendMsg 发往调度器
