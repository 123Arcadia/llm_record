import time
import tiktoken
import torch
import torch.nn as nn
from thop import profile


#####################################
# Chapter 3  无kv cache的实现
#####################################
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False
        )
        ###############################
        self.register_buffer("k_cache", None, persistent=False)
        self.register_buffer("v_cache", None, persistent=False)
        self.ptr_current_pos = 0
        ###############################

    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        values = self.W_value(x)
        queries = self.W_query(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        #################kv_cache####################
        if use_cache:
            if self.k_cache is None:
                self.k_cache, self.v_cache = keys, values
            else: # 第二维度num_tokens进行cat
                self.k_cache = torch.cat([self.k_cache, keys], dim=1)
                self.v_cache = torch.cat([self.v_cache, values], dim=1)
            keys, values = self.k_cache, self.v_cache
        else:
            keys, values = keys, values
        #############################################



        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        # mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        #################kv_cache####################
        num_token_Q = queries.shape[-2]
        num_token_K = keys.shape[-2]
        if use_cache:
            mask_bool = self.mask.bool()[ # 只考虑num_token_Q，num_token_K的loss计算
                self.ptr_current_pos:self.ptr_current_pos + num_token_Q, :num_token_K
            ]
            self.ptr_current_pos += num_token_Q
        else:
            mask_bool = self.mask.bool()[:num_token_Q, :num_token_K]
        #################kv_cache####################


        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec

    def reset_cache(self):
        self.k_cache, self.v_cache = None, None
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

class RMSNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(x.pow(2).mean()+self.eps)
        x /= rms
        x = self.gamma * x + self.beta
        return x



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
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )

    def forward(self, x):
        return self.layers(x)


class TransoformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        # self.norm1 = RMSNorm(cfg['emb_dim'])
        # self.norm2 = RMSNorm(cfg['emb_dim'])

        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        # x = self.attn(x)
        #################kv_cache####################
        # 传入use_cache参数
        x = self.attn(x, use_cache=use_cache)
        #################kv_cache####################
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
        #################kv_cache####################
        self.trf_blocks = nn.ModuleList(
            [TransoformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )
        self.current_pos = 0
        #################kv_cache####################
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx, use_cache=False):
        b, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx) #[b, seq_len, emb_dim]
        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        #################kv_cache####################
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else: # seq_len是可变的
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        #################kv_cache####################
        x = tok_embeds + pos_embeds  # [b, seq_len, emb_dim]
        # print(x.shape) # torch.Size([1, 0...203, 768])
        x = self.drop_emb(x)
        # x = self.trf_blocks(x)
        #################kv_cache####################
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        #################kv_cache####################
        x = self.final_norm(x)
        return self.out_head(x)  # 即logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.attn.reset_cache()
        self.current_pos = 0


def generate_text_sample(model, idx, max_new_tokens, context_size):
    # model.eval()
    for _ in range(max_new_tokens):

        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        idx_next = torch.argmax(logits,dim=-1, keepdim=True) # [bsz, 1]
        # print(f'{idx_cond.shape=}, {idx_next.shape=}')
        # idx_cond.shape=torch.Size([1, 203]), idx_next.shape=torch.Size([1, 1])
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_text_sample_cached(model, idx, max_new_tokens, context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings # num_embeddings是Embeddings层的自身数据，对应'vocab_size'
    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)
            for _ in range(max_new_tokens):
                # print(f'1->{idx.shape=}') # 1->idx.shape=torch.Size([1, 4...203])
                next_idx = logits[:, -1].argmax(dim=-1, keepdims=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True) # 把new_token输入
                # print(f'2->{idx.shape=} {next_idx.shape=}') # 2->idx.shape=torch.Size([1, 5...204]) next_idx.shape=torch.Size([1, 1])
        else:
            for _ in  range(max_new_tokens):
                # print(f'1->{idx.shape=}')
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdims=True)
                idx = torch.cat([idx, next_idx], dim=1)
                # print(f'2->{idx.shape=} {next_idx.shape=}')
    return idx


if __name__ == '__main__':
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-Key-Value bias
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    start_context = "Hello, I am"


    tokenizer = tiktoken.get_encoding('gpt2')
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0) # [b, seq_len]

    print(f"\n{50 * '='}\n{22 * ' '}IN\n{50 * '='}")
    print("\nInput text:", start_context)
    print("Encoded input text:", encoded)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()


    # token_ids = generate_text_sample(
    #     model=model,
    #     idx = encoded_tensor,
    #     max_new_tokens=200,
    #     context_size=GPT_CONFIG_124M['context_length']
    # )
    #################kv_cache####################
    token_ids = generate_text_sample_cached(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=200,
    )
    #################kv_cache####################




    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()
    total_time = end - start



    # # [b, seq_len, emb_dim] -> [seq_len, emb_dim]
    decoded_text = tokenizer.decode(token_ids.squeeze(0).tolist())

    print(f"\n\n{50 * '='}\n{22 * ' '}OUT\n{50 * '='}")
    print("\nOutput:", token_ids)
    print(f'Output shape: {token_ids.shape}') # [1, 204]
    print("Output length:", len(token_ids[0])) # 204
    print("Output text:", decoded_text) # ...

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")

    ######################################
    macs, params = profile(model, inputs=(encoded_tensor,), verbose=False)
    flops = 2 * macs
    print(f"{flops:.1e} FLOPS, params: {params / 1024 / 1024:.2f}")
    # 9.9e+08 FLOPS, params: 117.86
    ######################################


    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")

    # Time: 2.12 sec
    # 96 tokens/sec
    # Max memory allocated: 0.68 GB


    # 使用RMSNorm后
    # Time: 1.80 sec
    # 113 tokens/sec
    # Max memory allocated: 0.68 GB