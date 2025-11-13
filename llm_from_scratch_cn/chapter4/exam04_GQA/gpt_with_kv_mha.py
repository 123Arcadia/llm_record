import argparse

import thop
import torch
import torch.nn as nn
import time

import tiktoken


class MHA(nn.Module):
    """
    相比MHA_with_kvcache的没有`context_length`，故没有`self.mask`
    """

    def __init__(self, d_in, d_out, dropout, num_heads, qkv_bias=False):

        super().__init__()
        assert d_out % num_heads == 0, "维度和头数不能整除!"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)

        ##################kv_cache###################
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0
        ##################kv_cache###################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        k = self.W_key(x)
        q = self.W_query(x)
        v = self.W_value(x)

        #################第一种写法:先transpose在cachek和cachev#######################
        k_new = k.view(b, num_tokens, self.num_heads, -1).transpose(1, 2)
        v_new = v.view(b, num_tokens, self.num_heads, -1).transpose(1, 2)
        q = q.view(b, num_tokens, self.num_heads, -1).transpose(1, 2)

        #################第二种写法:先cachek和cachev在transpose在#######################
        # k_new = k.view(b, num_tokens, self.num_heads, -1).transpose(1, 2)
        # v_new = v.view(b, num_tokens, self.num_heads, -1).transpose(1, 2)
        # q_new = q.view(b, num_tokens, self.num_heads, -1).transpose(1, 2)
        #################第二种写法:先cachek和cachev在transpose在#######################

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k_new, v_new
            else:
                self.cache_k = torch.cat([self.cache_k, k_new], dim=2)
                self.cache_v = torch.cat([self.cache_v, v_new], dim=2)
            k, v = self.cache_k, self.cache_v
        else:
            k, v = k_new, v_new

        #################第二种写法:先cachek和cachev在transpose在#######################
        # k = k.transpose(1, 2)
        # v = v.transpose(1, 2)
        # q = q_new.transpose(1, 2)
        #################第二种写法:先cachek和cachev在transpose在#######################

        attn_scores = q @ k.transpose(2, 3)

        num_tokens_Q = q.shape[-2]
        num_tokens_K = k.shape[-2]
        # 要对num_tokens_Q的数量进行mask
        if use_cache:
            q_position = torch.arange(self.ptr_current_pos, self.ptr_current_pos + num_tokens_Q, device=x.device,
                                      dtype=torch.long)
            self.ptr_current_pos = self.ptr_current_pos + num_tokens_Q
        else:
            q_position = torch.arange(num_tokens_Q, device=x.device, dtype=torch.long)  # 从0开始
            self.ptr_current_pos = 0
        k_position = torch.arange(num_tokens_K, device=x.device, dtype=torch.long)
        # 当查询的位置小于键的位置，报名该位置需要屏蔽(设置为True)
        # 得到[num_tokens_Q, num_tokens_K]的mask
        # q:[num_tokens_Q, 1] - [1, num_tokens_K] -> [num_tokens_Q, num_tokens_K]
        mask_bool = q_position.unsqueeze(-1) < k_position.unsqueeze(0)

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, -1)
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
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim']),
        )
    def forward(self,x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MHA(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            dropout=cfg['drop_rate'],
            num_heads=cfg['n_heads'],
            qkv_bias = cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=True)
        x = self.drop_shortcut(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x += shortcut
        return x




class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-5
        self.gamma = 1.0

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(x.pow(2).mean() + self.eps)
        x = self.gamma * x / rms
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
        self.current_pos = 0 # 建立对pos_emb的cache

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx, use_cache=False):
        b, seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx) # [b, seq_len, emb_dim]
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)

        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0) # [b, 有效seq_len, emb_dim]
        x = tok_emb + pos_embeds
        x = self.drop_emb(x)

        for blk in self.trf_blocks:
            x = blk(x)

        x = self.final_norm(x)
        return self.out_head(x)

    def reset_cache(self):
        for b in self.trf_blocks:
            b.att.reset_cache()
        self.current_pos = 0
def generate_text_simple_cache(model, idx, max_new_tokens, context_length=None, use_cache=True):
    model.eval()
    ctx_len = context_length or model.pos_emb.num_embeddings # 维度
    with torch.no_grad():
        if use_cache:
            model.reset_cache()
            # 建立kv_cache
            logits = model(idx[:, -ctx_len:], use_cache=use_cache) # 因为x是[b, seq_len] ，所以logits也是

            for _ in range(max_new_tokens):
                next_idx = logits[:, -1].argmax(dim=-1, keepdims=True) # [b, 1]
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=use_cache)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=use_cache)
                next_idx = logits[:, -1].argmax(dim=-1, keepdims=True)
                idx = torch.cat([idx, next_idx], dim=1)

        return idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
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

    token_ids = generate_text_simple_cache(
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

    ##############################
    macs, param = thop.profile(model, inputs=(encoded_tensor,), verbose=True)
    print(f'{macs * 2:.1e} FLOPs, param: {param/1024/1024:.2f} M')

    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'总参数量:{count_params(model)/1024/1024:.2f}')
    print(f'out_head:{count_params(model.out_head)/1024/1024:.2f}')
    # 总参数量:154.86
    # out_head:36.81


    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")

    # [INFO] Register zero_ops() for <class 'torch.nn.modules.dropout.Dropout'>.
    # [INFO] Register count_linear() for <class 'torch.nn.modules.linear.Linear'>.
    # [INFO] Register zero_ops() for <class 'torch.nn.modules.container.Sequential'>.
    # 9.9e+08 FLOPS, param: 117.86
    #
    # Time: 2.29 sec
    # 89 tokens/sec
    # Max memory allocated: 0.33 GB


    #

# Output: tensor([[15496,    11,   314,   716, 49830,  5782, 24894,  2769, 33622, 15266,
#          14446,  8994,  2466,  5856, 45498, 46610, 43435, 19545, 39826, 21424,
#          29438, 25027, 43144, 30522, 32212, 24093, 49374, 13951, 40382, 48827,
#          18547, 25294, 21693, 47412, 14156,  6037, 48710,  9752,  4513, 40476,
#          30761, 47885, 39913,  5888, 25564, 18637, 33563, 17918, 12694, 37682,
#          45181, 28429, 11200, 26894, 20000,  3811, 10883, 35076, 46590, 40793,
#           4582, 26915, 12495, 14239, 26430, 42450, 33219, 36762,  9598, 48017,
#          34404, 11712, 17375, 48155, 40410, 29450, 39978,  2247, 33299, 19450,
#           2705, 48497,   653, 17151,  3999,  1513,  5182, 13975, 28497, 33091,
#           4740, 16430,  1029, 18379, 46132,  1843, 25904, 33501, 39555,  8433,
#           6755,  9207, 42336, 38305, 36273,  4417, 39689, 30756, 33494, 17669,
#          47947, 24830,   305,  5708,  4386, 48926, 33692, 22521, 29334,  6429,
#          36927, 37423, 40499, 26496, 26235,  2696,  5964, 23033, 32880,  1619,
#          19207, 13122, 26741, 38761,  3513, 50246,  9640,  2117,  7380, 16409,
#          44989,  2906,  8172, 39486, 25314, 38374, 15963, 42984,  6558, 48571,
#          25331, 50098, 15550, 11492, 14151,  8812,  7619, 22236, 26291,  7619,
#           9402,  2280, 18874, 38438, 39284,  9156, 47345, 13897, 30599, 44105,
#          35606, 31130, 14949, 24121, 34842, 17069, 26764, 41203, 31248, 15367,
#           8192,  3925,  6637, 12740, 43497,   882, 29158, 43128, 14564, 31521,
#          22767,  6023, 43515, 34215, 45406,  9205, 42417, 48135, 32536,  9108,
#          33649, 44074, 41655,  5724]], device='cuda:0')