# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# 类型注解工具
from typing import List

import torch
# 分词请求消息
from minisgl.message import TokenizeMsg
# HF tokenizer 基类
from transformers import PreTrainedTokenizerBase


# 分词管理器：把用户文本转成模型可用的 token id 张量
class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # 保存 tokenizer 实例
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        # 结果列表（与输入消息一一对应）
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                # 聊天式输入（dict 列表）：先套 chat template 拼成完整提示词；
                # add_generation_prompt=True 保证模板以助手回复的开头结尾
                prompt = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                # 模板渲染结果必须是字符串
                assert isinstance(prompt, str)
            else:
                # 补续式输入：文本即提示词
                prompt = msg.text
            input_ids: torch.Tensor = (  # type: ignore
                # 编码成 torch 张量（shape [1, seq]）
                self.tokenizer.encode(prompt, return_tensors="pt")
            )
            # 压平成 1 维并转 int32（调度器/KV 池统一用 int32 存 token id）
            results.append(input_ids.view(-1).to(torch.int32))
        # 返回每条请求的输入 token 张量
        return results
