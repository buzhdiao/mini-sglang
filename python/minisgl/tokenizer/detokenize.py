from dataclasses import dataclass  # dataclass 装饰器，用于声明 DecodeStatus 数据类
from typing import Dict, List  # 类型注解：Dict（uid->状态映射）与 List（列表）

from minisgl.message import DetokenizeMsg  # detokenize 请求消息：携带 uid、新 token id、是否结束等
from transformers import PreTrainedTokenizerBase  # HuggingFace 分词器基类，提供 batch_decode 接口

# Borrowed from sglang  # 注：本文件的流式可打印文本逻辑借鉴自 sglang 项目


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""  # 判断给定码点是否属于 CJK 中日韩字符区
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    # 上述英文注释：CJK 区块并非覆盖全部日文假名/韩文谚文（它们属于其他区块），这些语言按普通文本处理
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)  # CJK 统一表意文字基本区
        or (cp >= 0x3400 and cp <= 0x4DBF)  # CJK 统一表意文字扩展 A 区
        or (cp >= 0x20000 and cp <= 0x2A6DF)  # CJK 扩展 B 区
        or (cp >= 0x2A700 and cp <= 0x2B73F)  # CJK 扩展 C 区
        or (cp >= 0x2B740 and cp <= 0x2B81F)  # CJK 扩展 D 区
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  # CJK 扩展 E 区
        or (cp >= 0xF900 and cp <= 0xFAFF)  # CJK 兼容表意文字区
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  # CJK 兼容表意文字补充区
    ):  # 任一区间命中即为 CJK 字符
        return True

    return False  # 不在以上区块内，返回 False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""  # 返回 text 中只包含"完整单词"的最长可打印前缀
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    # 遇到换行符：换行天然是安全的输出边界，直接全部输出
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    # 最后一个字符是 CJK：中文无空格分词，逐字输出是安全的
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    # 倒数第二个字符是 CJK：最后一个字符可能是英文单词开头，先扣掉它
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    # 其他情况：只输出到最后一个空格为止——简单启发式，避免流式输出中打出一半、后续可能变化的"半截单词"
    else:
        return text[: text.rfind(" ") + 1]  # rfind 找不到空格返回 -1，切片变为 [:0]，即暂不输出


@dataclass  # 数据类装饰器，自动生成 __init__ 等方法
class DecodeStatus:
    """单个请求（uid）的增量解码状态：记录已解码内容与各阶段游标"""

    decoded_ids: List[int]  # 该请求累计收到的所有 token id（可能被 EOS 截断）
    decoded_str: str  # decoded_ids 全量解码出的字符串（稳定版文本）
    read_offset: int  # length of read ids  # 已"确认可读"的 token 数游标（即 surr 区间的右边界）
    surr_offset: int  # length of surr ids  # 已"稳定输出"的 token 数游标（read 区间的左边界）
    sent_offset: int  # length of sent out string  # 已发给前端的字符串长度游标（用于计算增量）


class DetokenizeManager:
    """detokenize 管理器：把调度器发来的新 token id 增量解码成文本，并保证流式输出不出现乱码/半截词"""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        # uid -> DecodeStatus
        self.decode_map: Dict[int, DecodeStatus] = {}  # 每个活跃请求的解码状态表，结束时删除
        self.tokenizer = tokenizer  # 分词器实例，用于 batch_decode
        self.eos_token_id = self.tokenizer.eos_token_id  # 结束符 id，用于在 finish 时丢弃末尾的 EOS token

    def detokenize(self, msgs: List[DetokenizeMsg]) -> List[str]:
        """处理一批 detokenize 消息，返回每条消息对应的增量输出字符串"""
        read_ids: List[List[int]] = []  # 每条消息"从稳定点开始"的全部 token id（可能包含未稳定的尾部）
        surr_ids: List[List[int]] = []  # 每条消息上一轮已稳定的 token id（作为前后缀对齐基准）
        for msg in msgs:  # 第一阶段：收集每条消息的 token id 区间
            if msg.uid not in self.decode_map:  # 首次见到该请求，初始化解码状态
                self.decode_map[msg.uid] = DecodeStatus(
                    decoded_ids=[],
                    decoded_str="",
                    read_offset=0,
                    surr_offset=0,
                    sent_offset=0,
                )
            s = self.decode_map[msg.uid]  # 取出该请求的解码状态
            if not (msg.finished and msg.next_token == self.eos_token_id):  # 结束且是 EOS 时不再追加（EOS 不输出给用户）
                s.decoded_ids.append(msg.next_token)  # 把本轮新 token id 追加进累计列表
            read_ids.append(s.decoded_ids[s.surr_offset :])  # 待解码区间：从稳定点到现在（含未稳定尾部）
            surr_ids.append(s.decoded_ids[s.surr_offset : s.read_offset])  # 上一轮已稳定的前缀区间（用于对齐截取）

        read_texts = self.tokenizer.batch_decode(read_ids)  # 批量解码"当前全量"文本
        surr_texts = self.tokenizer.batch_decode(surr_ids)  # 批量解码"已稳定前缀"文本

        incremental_strs: List[str] = []  # 最终返回的增量字符串列表
        for msg, read_str, surr_str in zip(msgs, read_texts, surr_texts, strict=True):  # 严格一一对应遍历
            s = self.decode_map[msg.uid]  # 取出该请求的解码状态
            new_text = read_str[len(surr_str) :]  # 去掉与上一轮稳定前缀的重叠部分，得到本轮真正新增的文本
            # Streaming chunk: update the decode status
            # 新增文本非空且不以 U+FFFD（多字节 UTF-8 字符被截断的替换符）结尾：说明 token 边界完整，可整体确认稳定
            if len(new_text) > 0 and not new_text.endswith("\ufffd"):
                output_str = s.decoded_str + new_text  # 累计成新的完整字符串
                s.decoded_str = output_str  # 更新稳定字符串
                s.surr_offset = s.read_offset  # 上一轮的 read 游标晋升为稳定游标
                s.read_offset = len(s.decoded_ids)  # read 游标推进到当前 token 总数
            else:
                # 文本为空或末尾有多字节字符被截断：本轮回退为"只输出可安全打印的部分"，游标不推进，等下一轮补齐
                new_text = find_printable_text(new_text)  # 截到最后一个空格/换行/CJK 边界，避免输出半截内容
                output_str = s.decoded_str + new_text  # 输出串 = 历史稳定串 + 本次可打印部分

            incremental_output = output_str[s.sent_offset :]  # 用"已发送长度"游标切出真正要推给前端的增量
            s.sent_offset = len(output_str)  # 更新已发送游标
            incremental_strs.append(incremental_output)  # 收集增量输出
            if msg.finished:  # 请求已结束，清理该 uid 的解码状态，防止内存泄漏
                del self.decode_map[msg.uid]

        return incremental_strs  # 返回与 msgs 一一对应的增量字符串
