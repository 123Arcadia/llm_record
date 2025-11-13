import torch
from torch import nn


# 定义一个带 dropout 的因果自注意力层
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, block_size, dropout, qkv_bias=False):
        '''
        构造函数，输入参数如下：
        d_in: 输入的维度
        d_out: 输出的维度
        block_size: 注意力权重矩阵的大小
        dropout: dropout 比例
        qkv_bias: 是否对 query、key 和 value 加偏置
        '''
        super().__init__()
        self.d_out = d_out
        # 根据前文，每一个权重矩阵都是 d_in x d_out 的线性层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 一个 dropout 层
        self.dropout = nn.Dropout(dropout)
        # 一个掩码矩阵，下三角为 1，其余为 0
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1)) # New

    def forward(self, x):
        '''
        前向传播函数，输入参数为 x，维度为 b x num_tokens x d_in，输出维度为 b x num_tokens x d_out
        '''
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        # transpose 是为了实现矩阵乘法
        attn_scores = queries @ keys.transpose(1, 2)
        # 即上文说过的，将掩码从 0 修改为 -inf，再进行遮蔽操作
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        # 经过 softmax
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
        # 进行 dropout
        attn_weights = self.dropout(attn_weights) # New
        # 得到最后结果
        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
            # 将 num_heads 个单头注意力层组合在一起来实现多头
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, block_size, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        # 前向计算时将多个头的输出拼接在一起
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, block_size, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 因为要对权重矩阵按注意力头数进行拆分，所有输出维度必须是头数的整数倍
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # head_dim 就是拆分之后每个头应该输出的维度
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(block_size, block_size), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # 形状为 (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 我们可以通过增加一个 num_heads 的维度来将矩阵分割到每个头
        # 维度变化: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 转置一下: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力权重
        # 基于矩阵乘法，简单地实现各个头的并行计算
        attn_scores = queries @ keys.transpose(2, 3)
        # 一般来说我们会将掩码矩阵转化为 bool 值并基于序列的长度进行截断
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 需要将掩码矩阵 unsqueeze 两次，也就是增加两个维度，才能让掩码矩阵的维度和注意力权重对应上
        mask_unsqueezed = mask_bool.unsqueeze(0).unsqueeze(0)
        # 使用掩码矩阵来进行遮蔽
        attn_scores.masked_fill_(mask_unsqueezed, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 形状: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 将多个头的输出重新组合回去 self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec