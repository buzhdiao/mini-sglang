from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许注解中使用未定义类型（如字符串形式的前向引用）
from dataclasses import dataclass  # dataclass 装饰器：自动生成 __init__/__repr__ 等
from typing import Any, Dict  # 类型注解：Any（任意类型）与 Dict（字典）
from transformers import PretrainedConfig  # HuggingFace 的模型配置基类：config.json 反序列化后的对象，携带全部模型超参


@dataclass(frozen=True)  # frozen=True：实例不可变（属性赋值会抛异常），保证配置在运行期不被意外修改
class RotaryConfig:
    """RoPE 旋转位置编码配置：描述位置编码的全部所需参数。"""

    head_dim: int  # 每个注意力头的维度（Q/K 向量长度）
    rotary_dim: int  # 参与旋转的维度数（通常等于 head_dim，即整个头维度都做旋转）
    max_position: int  # 模型支持的最大位置长度（如 4096/32768），用于预计算位置编码表
    base: float  # RoPE 的 theta 基数（如 10000/1000000），控制频率衰减速度
    scaling: Dict[str, Any] | None  # RoPE 外推/内插扩展策略（如 linear/YaRN），None 表示不缩放


@dataclass(frozen=True)  # 不可变数据类，同上
class ModelConfig:
    """模型结构配置：从 HF config.json 提取的本引擎所需的精简超参集合。

    命名对齐 vLLM/SGLang 惯例：qo=query/output 头，kv=key/value 头（支持 GQA 时 kv 头更少）。
    """

    num_layers: int  # Transformer 解码器层数（hidden_layers）
    num_qo_heads: int  # Q 头数量（注意：GQA 下 O 投影不改变维度，此字段沿用 Q 头数）
    num_kv_heads: int  # KV 头数量（GQA/MLA 中小于 Q 头数，用于省 KV Cache 显存）
    head_dim: int  # 每头维度，通常 = hidden_size // num_attention_heads
    hidden_size: int  # 隐藏层维度（词向量宽度），如 4096
    vocab_size: int  # 词表大小（embedding 行数）
    intermediate_size: int  # 稠密 MLP 的中间层宽度（FFN 隐藏维度）
    rms_norm_eps: float  # RMSNorm 的数值稳定 epsilon（防止除零）
    rotary_config: RotaryConfig  # 旋转位置编码配置（见上）
    hidden_act: str  # 激活函数名（"silu"/"gelu"），决定 MLP 用 silu_and_mul 还是 gelu_and_mul
    tie_word_embeddings: bool  # 是否共享输入 embedding 与输出 lm_head 权重（小模型常为 True）
    num_experts: int  # MoE：专家总数（稠密模型为 0）
    num_experts_per_tok: int  # MoE：每个 token 激活的专家数（top-k 路由）
    moe_intermediate_size: int  # MoE：每个专家 FFN 的中间层宽度
    norm_topk_prob: bool  # MoE：是否对 top-k 路由权重做归一化（和为 1）
    model_type: str  # HF config 的 model_type 字段（如 "llama"、"qwen3_moe"），用于判断是否 MoE
    architectures: list[str]  # HF config 的 architectures 字段（如 ["LlamaForCausalLM"]），用于查注册表

    @property  # 只读属性，像字段一样访问
    def is_moe(self) -> bool:
        # 通过 model_type 名是否含 "moe" 判断是否混合专家模型（如 qwen3_moe）
        return "moe" in self.model_type

    @classmethod  # 类方法：替代构造器，从 HF 原生配置对象转换为本引擎的 ModelConfig
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        """从 HuggingFace PretrainedConfig 提取本引擎需要的超参，构造 ModelConfig。

        处理了多模态模型（text_config 嵌套）、缺省字段兜底、Mistral 的 rope_theta 藏在
        rope_scaling 字典里等兼容性问题。
        """
        # 多模态模型（如 Llama-3.2-Vision）的真实文本模型配置嵌套在 text_config 中
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config  # 保存顶层（多模态整体）配置的引用
            config = config.text_config  # 切换到内层文本模型配置
            # 文本子配置可能缺少这些字段，从顶层配置回填（架构名/rope 参数）
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))  # 子配置缺该字段而顶层有时，把顶层值拷过来

        # GQA 兼容：老模型可能没有 num_key_value_heads 字段，此时 KV 头数 = Q 头数（MHA）
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        # 头维度：新模型（Qwen3 等）直接给 head_dim；老模型需要计算 hidden_size // 头数
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)  # 默认不共享 embedding/lm_head
        model_type = getattr(config, "model_type", "llama")  # 缺省按 llama 处理
        # 专家数兼容：Llama 系 MoE 用 num_local_experts，Qwen 系用 num_experts，都没有则 0（稠密模型）
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)  # 每 token 激活专家数，缺省 0
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)  # MoE 专家 FFN 宽度，缺省 0
        norm_topk_prob = getattr(config, "norm_topk_prob", False)  # 路由权重是否归一化
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])  # 模型类名，用于注册表分发

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        # Llama/Qwen 系：rope_theta 是直接属性；Mistral：把它藏进 rope_scaling 字典
        rope_scaling = getattr(config, "rope_scaling", None)  # RoPE 缩放配置（None 表示不缩放）
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]  # 优先直接属性，否则从 scaling 字典取（Mistral 风格）

        return cls(  # 用提取的值构造不可变的 ModelConfig
            num_layers=config.num_hidden_layers,  # HF 字段名：num_hidden_layers
            num_qo_heads=config.num_attention_heads,  # HF 字段名：num_attention_heads
            num_kv_heads=num_kv_heads,  # 上面兜底计算过的 KV 头数
            head_dim=head_dim,  # 上面兜底计算过的头维度
            hidden_size=config.hidden_size,  # 隐藏层宽度
            vocab_size=config.vocab_size,  # 词表大小
            intermediate_size=config.intermediate_size,  # 稠密 MLP 宽度
            hidden_act=config.hidden_act,  # 激活函数名
            rms_norm_eps=config.rms_norm_eps,  # RMSNorm eps
            tie_word_embeddings=tie_word_embeddings,  # 是否绑定词嵌入
            rotary_config=RotaryConfig(  # 嵌套构造 RoPE 配置
                head_dim=head_dim,  # 头维度
                rotary_dim=head_dim,  # 旋转维度=头维度（全维旋转）
                max_position=config.max_position_embeddings,  # 最大位置长度
                base=rope_theta,  # RoPE 基数 theta
                scaling=rope_scaling,  # RoPE 缩放策略
            ),
            num_experts=num_experts,  # 专家总数
            num_experts_per_tok=num_experts_per_tok,  # top-k 路由
            moe_intermediate_size=moe_intermediate_size,  # 专家 FFN 宽度
            norm_topk_prob=norm_topk_prob,  # 路由权重归一化开关
            model_type=model_type,  # 模型类型（is_moe 判断依据）
            architectures=architectures,  # 模型类名（注册表 key）
        )
