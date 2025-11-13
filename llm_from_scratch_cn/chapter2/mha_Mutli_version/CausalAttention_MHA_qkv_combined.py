import torch
from torch import nn

"""
这里qkv使用一个举证表达
在内部进行拆分q,k,v
"""
class MHA_qkv_Combined(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, f"维度和头数不能整除!{d_out=} {num_heads=}"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 每个attn的维度

        self.dropout = nn.Dropout(dropout)
        # self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        # self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        # self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.qkv = nn.Linear(d_in, 3 * d_out, bias=qkv_bias)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x):
        b, num_tokens, d_in = x.shape # [8, 1024, 768]

        # k = self.Wk(x)
        # q = self.Wq(x)
        # v = self.Wv(x)
        # print(f'{k.shape=}')
        # # k.shape=torch.Size([8, 1024, 768])
        #
        # k = k.view(b, num_tokens, self.num_heads, self.head_dim)  # head_dim=d_out//num_heads
        # q = q.view(b, num_tokens, self.num_heads, self.head_dim)
        # v = v.view(b, num_tokens, self.num_heads, self.head_dim) # 8 12 1024 64
        #
        # # qk^T
        # k = k.transpose_(1, 2)
        # q = q.transpose_(1, 2)
        # v = v.transpose_(1, 2)  # [b, num_heads, num_tokens, head_dim]
        # # print(f'{q.shape=}')
        # # q.shape=torch.Size([8, 1024, 12, 64])
        #
        # # Compute scaled dot-product attention (aka self-attention) with a causal mask
        # attn_scores = q @ k.transpose(2, 3)

        qkv = self.qkv(x)
        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim)
        # 把'3'提前
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0) # 按照指定维度拆分
        attn_scores = q @ k.transpose(2, 3)


        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # print(f'{attn_scores.shape=}') # 应该是[8, 12, 1024, 1024]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # attn_weights:[b, num_heads, num_tokens, head_dim]
        # attn_weights @ v:[b, num_tokens, num_heads, head_dim]
        # 经过transpose: [b, num_heads, num_tokens, head_dim]
        context_vec = (attn_weights @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")

    torch.manual_seed(123)
    batch_size = 8
    context_len = 1024
    embed_dim = 768
    embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)
    print(f'{embeddings.shape=}')
    # [8, 1024, 768]
    heads = 12

    # 12个头
    mha = MHA_qkv_Combined(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=context_len,
        dropout=0.0,
        num_heads=heads,
        qkv_bias=False
    ).to(device)
    out = mha(embeddings)
    print(f'{out.shape=}')
    # out.shape=torch.Size([8, 1024, 768])
