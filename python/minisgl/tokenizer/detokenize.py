# dataclass 生成解码状态类
from dataclasses import dataclass
# 类型注解工具
from typing import Dict, List

# 反解码消息（uid + 下一个 token + 是否结束）
from minisgl.message import DetokenizeMsg
# HF tokenizer 基类
from transformers import PreTrainedTokenizerBase

# Borrowed from sglang
# 以下流式反解码工具借鉴自 sglang


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    # 判断码点是否落在各 CJK 统一表意文字区段内
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        # 命中任一区段即为 CJK 字符
        return True

    # 其余码点不是 CJK 字符
    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    # 以换行结尾：整行内容必定完整，直接全部输出
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    # 末字符是 CJK：CJK 无空格分词，每个字符可独立成词，直接全部输出
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    # 倒数第二字符是 CJK：末尾那个字符可能是多字节词的开头，先扣掉它
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    # 其他情况：只输出到最后一个空格为止——英文单词可能被拆在多个 token 里，
    # 后续 token 会改变当前"半个词"的解码结果，先不发以免发出去又变
    else:
        return text[: text.rfind(" ") + 1]


# 一条请求的流式解码状态
@dataclass
class DecodeStatus:
    # 该请求到目前为止收到的全部 token id
    decoded_ids: List[int]
    # 已确认稳定的解码文本
    decoded_str: str
    read_offset: int  # length of read ids
    # 已送入解码器（但可能还不稳定）的 token 数
    surr_offset: int  # length of surr ids
    # 已确认稳定（上次作为前缀）的 token 数
    sent_offset: int  # length of sent out string
    # 已发送给用户的文本长度


# 反解码管理器：把 token id 流增量地转成文本流（处理多字节/部分词的稳定性问题）
class DetokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        # 每条在生成请求的状态表
        self.decode_map: Dict[int, DecodeStatus] = {}
        # 保存 tokenizer
        self.tokenizer = tokenizer
        # 预取 eos id（结束 token 不进入输出文本）
        self.eos_token_id = self.tokenizer.eos_token_id

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        # 每条消息对应的"已读未确认"token 列表（批量解码用）
        read_ids: List[List[int]] = []
        # 每条消息对应的"已确认稳定"前缀 token 列表
        surr_ids: List[List[int]] = []
        for msg in msgs:
            if msg.uid not in self.decode_map:
                # 首次见到该请求：初始化空的解码状态
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            # 取该请求的状态
            s = self.decode_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                # 结束且生成的是 eos 时不追加：eos 只是终止信号，不属于输出内容
                s.decoded_ids.append(msg.next_token)
            # 从上次稳定点开始的所有 token（包含可能变化的不稳定尾部）
            read_ids.append(s.decoded_ids[s.surr_offset :])
            # 上次稳定点到上次已读点之间的 token（本步之前已读过的部分）
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])

        # 批量解码：一次调用同时解出所有请求的文本（利用 HF 的批处理）
        read_texts = self.tokenizer.batch_decode(read_ids)
        # 批量解码稳定前缀（用于对比求增量）
        surr_texts = self.tokenizer.batch_decode(surr_ids)

        # 每条消息对应的增量输出文本
        incremental_strs: List[str] = []
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):
            # 取该请求状态
            s = self.decode_map[msg.uid]
            # 本步真正新增的文本 = 重新解码的全串 - 稳定前缀串
            new_text = read_str[len(surr_str) :]
            # Streaming chunk: update the decode status
            # 新文本非空且不以 U+FFFD 结尾：末尾不是被截断的多字节字符，
            # 整段可以确认为稳定——推进全部偏移
            if len(new_text) > 0 and not new_text.endswith("�"):
                output_str = s.decoded_str + new_text
                # 更新已确认文本
                s.decoded_str = output_str
                # 旧的已读边界晋升为新的稳定边界
                s.surr_offset = s.read_offset
                # 已读边界推进到当前 token 总数
                s.read_offset = len(s.decoded_ids)
            else:
                # 文本不稳定（半个多字节字符/半个词）：只挑出"肯定不会再变"的
                # 部分输出（到换行/CJK/空格为止），剩余部分留给下一轮
                new_text = find_printable_text(new_text)
                # 拼接到输出串（偏移不推进：不稳定尾部下轮重新解码）
                output_str = s.decoded_str + new_text

            # 本轮发给用户的增量 = 完整输出 - 上次已发送前缀
            incremental_output = output_str[s.sent_offset :]
            # 更新已发送边界
            s.sent_offset = len(output_str)
            # 收集增量
            incremental_strs.append(incremental_output)
            if msg.finished:
                # 请求结束：清理该请求的解码状态
                del self.decode_map[msg.uid]

        # 返回每条请求的增量文本
        return incremental_strs
