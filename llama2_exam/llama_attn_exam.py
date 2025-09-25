from typing import Tuple, Optional

import torch
import math
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"

    def __init__(
            self,
            dim: int = 768,  # 模型维度
            n_layers: int = 12,  # Transformer的层数
            n_heads: int = 16,  # 注意力机制的头数
            n_kv_heads: int = 8,  # 键值头的数量
            vocab_size: int = 6144,  # 词汇表大小
            hidden_dim: int = None,  # 隐藏层维度
            multiple_of: int = 64,
            norm_eps: float = 1e-5,  # 归一化层的eps
            max_seq_len: int = 512,  # 最大序列长度
            dropout: float = 0.0,  # dropout概率
            flash_attn: bool = True,  # 是否使用Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        # 可学习的参数，初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    扩展和重塑张量, 使kv和Q的维度一致
    :param x:
    :param n_rep:
    :return:
    """
    bs, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x

    return (x[:, :, :, None, :]
            .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(bs, seq_len, n_kv_heads * n_rep, head_dim))


# 旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """

    :param dim: 指的是每个head的dim
    :param end:
    :param theta:
    :return:
    """
    # 生成了一个从0开始，步长为2的序列，其长度为dim的一半。
    # 每个元素除以dim后取theta的倒数，得到一个频率序列 freqs。这一步是为了生成适合旋转嵌入的频率。
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 0到end序列
    t = torch.arange(0, end, device=freqs.device)
    # 外积
    freqs = torch.outer(t, freqs).float()
    freq_cos = torch.cos(freqs)  # 实部
    freq_sin = torch.sin(freqs)  # 虚部
    return freq_cos, freq_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor,  # [50, 24]
                          x: torch.Tensor):  # 1, 50, 6, 24
    """
    调整 freqs_cis 的形状，使其在进行广播操作时与 x 的维度对齐，从而能够进行正确的张量运算
    :param freqs_cis:
    :param x:
    :return:
    """
    ndim = x.ndim
    # 确保1在ndim内
    assert 0 <= 1 < ndim
    #
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 除了第二维和最后一维，其他维度都为1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    # [1, 50, 1, 24]
    return freqs_cis.view(shape)


def apply_rotary_emb(xq: torch.Tensor,  # [1, 50, 6, 48]
                     xk: torch.Tensor,
                     freqs_cos: torch.Tensor,  # [50, 24]
                     freqs_sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # 将q, k转换为float
    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    # xq: bs, seq_len, dim//n_head, n_head_dim -> sh=torch.Size([1, 50, 6, -1, 2]) -> unbind ->
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)
    # print(f'{xq_r.shape=}  {xq_i.shape=}')
    # xq_r.shape=torch.Size([1, 50, 6, 24])  xq_i.shape=torch.Size([1, 50, 6, 24])

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)
    # shape= [1, 50, 1, 24]

    # 应用旋转，分别计算旋转后的实部和虚部
    # print(f'{(xq_r * freqs_cos).shape=}')
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    # print(f"{xq_out_r.shape=}")
    # xq_out_r.shape=torch.Size([1, 50, 6, 24])

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
    # print(f"{xq_out.shape=}")
    # xq_out.shape=torch.Size([1, 50, 6, 48])

    return xq_out.type_as(xq), xk_out.type_as(xk)


class llama_Attn(nn.Module):
    """
    实现MQA
    """

    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行数
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵。
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # 输出权重矩阵。dim * dim
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # 是否使用flash attn
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)  # 包含主对角线的上三角都是0
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
        # [batch_size, seq_len, dim]
        bsz, seq_len, _ = x.shape

        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
        # 对KQ扩展
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            # 手动attn
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, seq_len, seq_len]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        # 然后将其减少到2/3，最后确保它是multiple_of的倍数
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim * multiple_of - 1) // multiple_of)

        # print(f'{__name__} : {hidden_dim=}')
        # llama_attn_exam : hidden_dim=131008

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // self.n_heads
        self.attn = llama_Attn(args)
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        # 对attn和fnn来说都是Pre-Norm
        self.attn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        # x -> 注意力归一化 -> 与x相加 得到 h
        # h -> 前馈网络归一化 -> feed_ward -> 与h相加
        h = x + self.attn.forward(self.attn_norm(x), freqs_cos, freqs_sin)
        o = h + self.feed_forward.forward(self.ffn_norm(h))
        return o


class Transformer(PreTrainedModel):
    config_class: ModelConfig  # 配置类
    last_loss: Optional[torch.Tensor]  # 记录最后一次计算的损失

    def __init__(self, args: ModelConfig = None):
        super().__init__(args)
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers

        self.tok_embdings = nn.Embedding(args.vocab_size, args.dim)
        self.dropout = nn.Dropout(args.dropout)
        self.layers = torch.nn.ModuleList()
        for l_id in range(args.n_layers):
            self.layers.append(DecoderLayer(l_id, args))
        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

        # 将词嵌入层的权重与输出层的权重共享
        self.tok_embdings.weight = self.output.weight

        # 预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # 初始化权重
        self.apply(self._init_weights)
        # 对所有残差投影进行缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * args.n_layers))

        # 初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器
        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        """
       - tokens: Optional[torch.Tensor], 输入 token 张量。
       - targets: Optional[torch.Tensor], 目标 token 张量。
       - kv_cache: bool, 是否使用键值缓存。
       - kwargs: 其他关键字参数。

       - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
       """
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']

        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']

        _bsz, seq_len = tokens.shape
        h = self.tok_embdings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]

        #通过decoder层
        for l in self.layers:
            h = l(h ,freqs_cos, freqs_sin)

        h = self.norm(h)

        if targets is not None:
            # 计算loss
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            # 推理时的小优化：只对最后一个位置的输出进行前向传播
            logits = self.output(h[:, [-1], :]) #选择最后一个seq_len
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode
    def generate(self, idx: torch.Tensor, stop_id=None, max_new_token=256, temperature = 1.0, top_k=None):
        """
            给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
            在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
        """
        index = idx.shape[1]
        for _ in range(max_new_token):
            # 如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(-1) <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            # logits是模型的原始未归一化的分数
            logits = self(idx_cond).logits
            # 前向传播获取序列中最后一个位置的 logits
            # logits维度: [bdz, seq_len, vocab_size] bsz, 生成的token长度, 词汇表大小
            logits = logits[:, -1, :]  # 只保留最后一个时间步的输出

            if temperature == 0.0:
                # 选择最有可能的索引
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # 缩放 logits 并应用 softmax
                logits = logits / temperature
                # 总体就是先把topk 中大于 logits的全部设置为-inf，这些在softmax后0
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1) # 在剩下这些softmax不为0的数中，对其分布概率进行采样
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break

            # 将采样的索引添加到序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)
            return idx[:, index:] # 只返回生成的token



if __name__ == '__main__':
    # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
    x = torch.randint(0, 6144, (1, 50))  # [bs, seq_len]
    # 实例化LLaMA2Model
    args = ModelConfig()
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)
    # Number of parameters: 3648080640

    out = model(x)
    print(out.logits.shape)  # [batch_size, 1, vocab_size]
    #torch.Size([1, 1, 6144])




























