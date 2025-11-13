import  torch
from torch import nn
import torch.nn.functional as F



class Scaled_dot_Attention(nn.Module):
    """
    torch内部scale_dot_attn实际是 FlashAttention实现
    """
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.0, qkv_bias = False):
        super().__init__()
        assert d_out % num_heads == 0, f"输入维度和头数不能整除!!!{d_out=}, {num_heads=}"
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = d_out // num_heads
        self.d_out = d_out

        self.proj = nn.Linear(d_out, d_out)
        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.dropout = dropout

    def forward(self, x):
        b, num_tokens, emb_dim = x.shape # emb_dim=d_int
        # print(f'{x.shape=}') # x.shape=torch.Size([8, 1024, 768])

        qkv = self.qkv(x)
        qkv = qkv.view(b, num_tokens, 3, self.num_heads, self.head_dim)
        # print(f'{qkv.shape=}') # qkv.shape=torch.Size([8, 1024, 3, 12, 64])
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        dropout = 0. if self.dropout is None else self.dropout
        # 得到[bs, num_heads, num_tokens, head_dim]
        context_vec = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True)

        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, -1)
        return self.proj(context_vec)


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
    mha = Scaled_dot_Attention(
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






