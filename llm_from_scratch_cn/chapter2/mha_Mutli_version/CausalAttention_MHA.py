import torch
from torch import nn

class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)  # New
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))  # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # print(f'{x.shape=}')
        # # x.shape=torch.Size([8, 1024, 768])
        # print(f'{keys.shape=}')
        # # keys.shape=torch.Size([8, 1024, 64])

        attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
        # print(f'{attn_scores.shape=}') #attn_scores.shape=torch.Size([8, 1024, 1024])

        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)  # New

        context_vec = attn_weights @ values
        return context_vec

"""
没有在Attn中进行分割维度，而是在输入中就是分开
"""
class Ch03_MHA_Wrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )
        self.out_proj = nn.Linear(d_out*num_heads, d_out*num_heads)

    def forward(self, x):
        context_vec = torch.cat([head(x) for head in self.heads], dim=-1)
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
    # embeddings.shape=torch.Size([8, 1024, 768])
    heads = 12

    # 12个头
    mha_ch03_wrapper = Ch03_MHA_Wrapper(
        d_in=embed_dim,
        d_out=embed_dim // heads,
        context_length=context_len,
        dropout=0.0,
        num_heads=heads,
        qkv_bias=False
    ).to(device)
    out = mha_ch03_wrapper(embeddings)
    print(f'{out.shape=}')
    # Using device: cuda
    # PyTorch version: 2.4.1+cu121
    # out.shape=torch.Size([8, 1024, 768])