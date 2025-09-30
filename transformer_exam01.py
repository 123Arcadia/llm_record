import torch
import math

from torch import nn
from dataclasses import dataclass
# from transformers import BertTokenizer
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM


@dataclass
class ModelArgs:
    n_emb: int  # 嵌入维度
    n_heads: int
    dim: int
    dropout: float
    max_seq_len: int
    vocab_size: int
    block_size: int  # 序列的最大长度
    n_layer: int


class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal=False):
        super().__init__()
        assert args.dim % args.n_heads == 0, "隐藏层dim不是head头数的整数被"
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        self.wq = nn.Linear(args.n_emb, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_emb, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_emb, self.n_heads * self.head_dim, bias=False)
        # 输出维度: dim * dim
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # attn的dropout
        self.attn_drop = nn.Dropout(args.dropout)
        # 残差d drop
        self.resid_drop = nn.Dropout(args.dropout)
        self.is_causal = is_causal
        # 上三角矩阵
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)  # 返回上三角(不包含主对角线)，其余为0
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # [batch_size, seq_len, dim]
        bsz, seq_len, _ = q.shape
        # 计算q,k, v -> [bsz, seq_len, dim] * [emb_dim, dim] = [bsz, seq_len, dim]
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)
        # attn中取后两个维度计算
        # 为什么先展开[bsz, seq_len, n_heads, head_dim] 然后在互换1,2维度?
        # view先全部排开，按照要求构造
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if self.is_causal:
            assert hasattr(self, "mask")
            scores = scores + self.mask[:, :, :seq_len, :seq_len]

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # dropout
        scores = self.attn_drop(scores)
        # 和xv: 维度: [bsz, n_head, seq_len, seq_len] * [bsz, n_head, seq_len, head_dim]
        # 得到:[bsz, n_head, seq_len, head_dim]
        output = torch.matmul(scores, xv)
        # 回复成[bsz, seq_len, dim]
        # transpose不会涉及底层存储，所以需要连续化
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

        # [[bsz, seq_len, dim]] * [dim, dim]
        output = self.wo(output)
        # 通过残差
        output = self.resid_drop(output)
        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """

        :param features: 即n_emb
        :param eps:
        """
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    # 缩放+平移
    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)  # [bsz, seq_len, 1]
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        # w1和w2两个线性层之间有一个relu激活函数
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))


class EncodeLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 一个layer中有两个layerNorm,在attn之前和MLP前
        self.attn_norm = LayerNorm(args.n_emb)
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_emb)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        x = self.attn_norm(x)
        # 自注意力 + 残差: 输入qkv
        h = x + self.attention.forward(x, x, x)
        o = h + self.feed_forward.forward(h)
        return o


class Encoder(nn.Module):
    """
    Enocder有多个EncoderLayer组成（每个EncoderLayer包含一个mha、两个layerNorm、和MLP）
    """

    def __init__(self, args: ModelArgs):
        # 为了兼容py2.x和py3.x
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncodeLayer(args) for _ in range(args.n_layer))
        self.norm = LayerNorm(args.n_emb)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_emb)
        self.mask_attn = MultiHeadAttention(args, is_causal=True)
        self.attn_norm_2 = LayerNorm(args.n_emb)
        self.attn = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_emb)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        x = self.attention_norm_1(x)
        # 掩码attn
        x = x + self.mask_attn.forward(x, x, x)
        # mha
        x = self.attn_norm_2(x)
        h = x + self.attn.forward(x, enc_out, enc_out)
        o = h + self.feed_forward.forward(self.ffn_norm(x))
        return o



class Decoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_emb)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)


class PositionalEncoding(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        pe = torch.zeros(args.block_size, args.n_emb) # [block_size, n_emb]
        pos = torch.arange(args.block_size).unsqueeze(1) # [block_size, 1]
        # theta
        div_term = torch.exp(torch.arange(0, args.n_emb, 2) * -(math.log(10000.0) / args.n_emb)) #应该是args.dim吧
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 位置编码加到emb结果上
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert  args.vocab_size is not None
        assert  args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(args.vocab_size, args.n_emb),
                wpe = PositionalEncoding(args),
                drop= nn.Dropout(args.dropout),
                encoder = Encoder(args),
                decoder = Decoder(args)
            )
        )
        # qkv后是[bdx, seq_len, dim]
        # 最后的线性层, 输入: n_emb, 输出： 词表大小
        self.lm_head = nn.Linear(args.n_emb, args.vocab_size, bias=False)
        # 初始化参数
        self.apply(self._init_weights)
        # 所有参数的数量
        # print(f'number of params:{self.get_num_params() / 1e6:%.2fM} %')
        print(f'number of params:{self.get_num_params() / 1e6} M')

    def get_num_params(self, non_embedding=False) ->float:
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params



    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx: torch.Tensor, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size() # [bsz, seq_len]
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"
        # 先idx通过emb层，得到[bsz, seq_len, n_emb]
        print(f'idx: {idx.size()}')
        tok_emb = self.transformer.wte(idx)
        print(f'tok_emb: {tok_emb.size()}')
        # 通过位置编码
        pos_emb = self.transformer.wpe(tok_emb)
        # dropout
        x = self.transformer.drop(pos_emb)
        print(f'x after wpe(x.size):{x.size()}')
        enc_out = self.transformer.encoder(x)
        print(f'{enc_out.size()=}')
        x = self.transformer.decoder(x, enc_out)
        print(f'x after decoder(x.size): {x.size()}')

        if targets is not None:
            # 训练是。如果给了target，就计算loss
            # 通过最后的linear层，得到[bsz, seq_len, vocab_size]
            logits = self.lm_head(x)
            # 在与target计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理是，只需要logits
            # 取 -1 是只取序列中的最后一个作为输出
            # [bzz, seq_len, vocab_size]
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

def main():
    args = ModelArgs(100, 10, 100, 0.1, 512, 1000, 1000, 2)
    text = "我喜欢快乐地学习大模型"
    # 输出：
    # start!!!
    # number of params:4.5488 M
    # idx: torch.Size([1, 512])
    # tok_emb: torch.Size([1, 512, 100])
    # x after wpe(x.size):torch.Size([1, 512, 100])
    # enc_out.size()=torch.Size([1, 512, 100])
    # x after decoder(x.size): torch.Size([1, 512, 100])
    # logits: tensor([[[ 0.0326, -0.0684, -0.4752,  ...,  0.0974, -0.1351,  0.2127]]],
    #        grad_fn=<UnsafeViewBackward0>)
    # output='##鹕'


    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    # tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    inputs_token = tokenizer(
        text,
        return_tensors='pt',
        max_length=args.max_seq_len,
        truncation=True,
        padding='max_length'
    )
    args.vocab_size = tokenizer.vocab_size
    transformer = Transformer(args)
    print(f'{text=}  {inputs_token.keys()=}')

    inputs_id = inputs_token['input_ids']
    logits, loss = transformer.forward(inputs_id)
    print(f'logits: {logits}')
    predicted_ids = torch.argmax(logits, dim=-1).item()
    output = tokenizer.decode(predicted_ids)
    print(f'{output=}')

if __name__ == "__main__":
    print('start!!!')
    main()

    # number of params:4.5488 M
    # text='你好'  inputs_token.keys()=KeysView({'input_ids': tensor([[ 101,  872, 1962,  102,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,    0,
    #             0,    0,    0,    0,    0,    0,    0,    0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0]])})
    # idx: torch.Size([1, 512])
    # tok_emb: torch.Size([1, 512, 100])
    # x after wpe(x.size):torch.Size([1, 512, 100])
    # enc_out.size()=torch.Size([1, 512, 100])
    # x after decoder(x.size): torch.Size([1, 512, 100])
    # logits: tensor([[[ 0.0789, -0.2442, -0.1967,  ...,  0.1644,  0.0995,  0.2239]]],
    #        grad_fn=<UnsafeViewBackward0>)
    # predicted_ids=3607
    # output='櫻'