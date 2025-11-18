import argparse
import time
import tiktoken
import torch
import torch.nn as nn
from thop import profile
from tqdm import tqdm


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dropout, dtype=None, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "输出维度和头数不能整除!"
        assert num_heads % num_kv_groups == 0, "总头数和分组数不能整除!"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # mha中是num_heads * self.head_dim, gqa中是num_kv_groups * self.head_dim
        # 总头数有分组数代替
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=qkv_bias, dtype=dtype)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)

        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # K: [b, self.num_kv_groups, num_tokens, , self.head_dim]
        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=2)
                self.cache_v = torch.cat([self.cache_v, v], dim=2)
            k, v = self.cache_k, self.cache_v
        else:
            k, v = k, v
            if self.cache_k is not None or self.cache_v is not None:
                self.reset_cache()

        k = k.repeat_interleave(self.group_size, dim=1)  # [b, group_size * num_kv_heads, num_tokens, head_dim]
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_scores = q @ k.transpose(2, 3)
        # 建立mask举证
        num_tokens_Q = q.shape[-2]  # 该prompt的输入的num_tokens长度
        num_tokens_K = k.shape[-2]
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if use_cache:
            q_position = torch.arange(self.ptr_current_pos, self.ptr_current_pos + num_tokens_Q, dtype=torch.long,
                                      device=device)
            self.ptr_current_pos += num_tokens_Q
        else:
            q_position = torch.arange(0, num_tokens_Q, dtype=torch.long, device=device)
            self.ptr_current_pos = 0
        k_position = torch.arange(num_tokens_K, dtype=torch.long, device=device)
        mask = q_position.unsqueeze(-1) < k_position.unsqueeze(0)

        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        assert k.shape[-1]==self.head_dim, "k 的最优应该维度 != head_dim"
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, -1) # [b, num_tokens, d_out]
        return self.out_proj(context_vec)

    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            num_heads=cfg['n_heads'],
            num_kv_groups=cfg['n_kv_groups'],
            qkv_bias=cfg['qkv_bias'],
            dropout=cfg['drop_rate']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut
        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.current_pos = 0  # 对应pos_emb
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_dix, use_cache=False):
        b, seq_len = in_dix.shape
        tok_embed = self.tok_emb(in_dix)
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_dix.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_dix.device, dtype=torch.long)
        pos_embed = self.pos_emb(pos_ids).unsqueeze(0) # [1, b, seq_len, emb_dim]
        x = tok_embed + pos_embed # 广播机制
        x = self.drop_emb(x)
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        x = self.final_norm(x)
        return self.out_head(x)

    def reset_kv_cache(self):
        for b in self.trf_blocks:
            b.att.reset_cache()
        self.current_pos = 0


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    if use_cache:
        model.reset_kv_cache()
        logits = model(idx[:, -ctx_len:], use_cache=use_cache)
        for _ in tqdm(range(max_new_tokens)):
            next_idx = logits[:, -1].argmax(dim=-1, keepdims=True)
            idx = torch.cat([idx, next_idx], dim=1)
            logits = model(next_idx, use_cache=use_cache)

    else:
        for _ in tqdm(range(max_new_tokens)):
            logits = model(idx[:, -ctx_len:], use_cache=use_cache)
            next_idx = logits[:, -1].argmax(dim=-1, keepdims=True)
            idx = torch.cat([idx, next_idx], dim=1) # dim=1要确定在seq_len哪一个维度上, -1会在d_in维度上

    return idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="gpt-kv-cache-gqa")

    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--n_kv_groups", type=int, default=2, help="Number of key/value groups.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,  # Embedding dimension
        "n_heads": args.n_heads,  # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
        "n_kv_groups": args.n_kv_groups
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    token_ids = generate_text_simple_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=args.max_new_tokens,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.time() - start

    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    # 计算flops、params
    macs, params = profile(model, inputs=(encoded_tensor,), verbose=True)
    print(f'GQA flops: {macs * 2:.1e}, params: {params / 1024/1024:.2f} M')

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")

    # ==================================================
    #                       IN
    # ==================================================
    #
    # Input text: Hello, I am
    # Encoded input text: [15496, 11, 314, 716]
    # encoded_tensor.shape: torch.Size([1, 4])
    #
    #
    # ==================================================
    #                       OUT
    # ==================================================
    #
    # Output: tensor([[15496,    11,   314,   716, 21658, 11486, 19510,  2290, 35604, 11218,
    #           1855, 23137,  3912, 14450, 22713,  4700, 37888, 47815, 11609,  7759,
    #          34461, 46798,  9646, 22415, 19643, 48892, 39385,  1353, 12087, 47059,
    #          20072, 30553, 19391, 45598, 34723, 21571, 48186, 36655, 31597, 35923,
    #          22402, 10743, 31534, 17941, 40799, 18492, 44978, 41938, 33489, 49416,
    #          12989, 43802, 37320, 21073, 32690,  9980,  1580, 11331, 35776, 23262,
    #          29750, 47514, 11721, 24160, 33885, 25457, 27851, 18475, 37746,  3166,
    #          44882,  1293, 27186, 12781, 30426, 28393, 10004,  3264, 27006,  9521,
    #          32090, 37063, 43770, 19768, 24786, 19656, 47913, 48705, 28009, 27882,
    #          37602,  1537, 41179,  2839, 15585, 14143, 49742, 17616, 31511,  3850,
    #          23152, 32146, 26546, 47508, 25774, 12721, 19221, 20392,  5075,  7727,
    #          34368, 21303, 39549, 13201, 25438, 28793, 37144, 17091, 47046, 10026,
    #          48752, 21144, 33079, 10144, 14018, 24786, 47783, 49911, 10795,  2807,
    #          31528,  3559, 24480, 35742, 22939, 47000, 12353, 36829, 32289, 43909,
    #          28963,  9962,  3589,  9063, 45606, 25515, 43272, 48470, 48058,  8451,
    #          18406, 17620, 38280, 47100, 28851, 15267,    21, 10889, 33639, 17572,
    #          13767, 20367, 32350, 27808, 47517, 29874, 40158, 33930, 10079, 20165,
    #          11697,  9441, 13369,  1486,  1826,  5884, 42269, 10712, 45939, 18698,
    #          22518, 27217, 50181, 11460, 47391, 34865, 42240, 41524, 27149,  5581,
    #          23751,   883,  5737, 42808, 18436, 25795, 24867,  7019, 49283, 50067,
    #          17553, 12086, 24570,  3788]], device='cuda:0')
    # Output length: 204

    # Output: tensor([[15496,    11,   314,   716, 21658, 11486, 19510,  2290, 35604, 11218,
    #           1855, 23137,  3912, 14450, 22713,  4700, 37888, 47815, 11609,  7759,
    #          34461, 46798,  9646, 22415, 19643, 48892, 39385,  1353, 12087, 47059,
    #          20072, 30553, 19391, 45598, 34723, 21571, 48186, 36655, 31597, 35923,
    #          22402, 10743, 31534, 17941, 40799, 18492, 44978, 41938, 33489, 49416,
    #          12989, 43802, 37320, 21073, 32690,  9980,  1580, 11331, 35776, 23262,
    #          29750, 47514, 11721, 24160, 33885, 25457, 27851, 18475, 37746,  3166,
    #          44882,  1293, 27186, 12781, 30426, 28393, 10004,  3264, 27006,  9521,
    #          32090, 37063, 43770, 19768, 24786, 19656, 47913, 48705, 28009, 27882,
    #          37602,  1537, 41179,  2839, 15585, 14143, 49742, 17616, 31511,  3850,
    #          23152, 32146, 26546, 47508, 25774, 12721, 19221, 20392,  5075,  7727,
    #          34368, 21303, 39549, 13201, 25438, 28793, 37144, 17091, 47046, 10026,
    #          48752, 21144, 33079, 10144, 14018, 24786, 47783, 49911, 10795,  2807,
    #          31528,  3559, 24480, 35742, 22939, 47000, 12353, 36829, 32289, 43909,
    #          28963,  9962,  3589,  9063, 45606, 25515, 43272, 48470, 48058,  8451,
    #          18406, 17620, 38280, 47100, 28851, 15267,    21, 10889, 33639, 17572,
    #          13767, 20367, 32350, 27808, 47517, 29874, 40158, 33930, 10079, 20165,
    #          11697,  9441, 13369,  1486,  1826,  5884, 42269, 10712, 45939, 18698,
    #          22518, 27217, 50181, 11460, 47391, 34865, 42240, 41524, 27149,  5581,
    #          23751,   883,  5737, 42808, 18436, 25795, 24867,  7019, 49283, 50067,
    #          17553, 12086, 24570,  3788]], device='cuda:0')


    # [INFO] Register zero_ops() for <class 'torch.nn.modules.dropout.Dropout'>.
    # [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
    # [INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
    # GQA flops: 8.9e+08, params: 106.60 M
    #
    # Time: 3.03 sec
    # 67 tokens/sec
    # Max memory allocated: 1.06 GB