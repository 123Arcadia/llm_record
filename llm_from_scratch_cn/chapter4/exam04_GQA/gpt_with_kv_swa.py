import argparse
import time
import tiktoken
import torch
import torch.nn as nn
from tqdm import tqdm

class MultiHeadAttentionWithSWA(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout, qkv_bias=False, sliding_window_size=None):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.sliding_window_size = sliding_window_size

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.ptr_current_pos = 0


    def forward(self, x, use_cache=False):
        b, num_tokens, d_in = x.shape

        q = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim)
        k = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim)
        v = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim)

        if use_cache:
            old_len = 0 if self.cache_k is None else self.cache_k.size(1)
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)

            if self.sliding_window_size is not None:
                if self.cache_k.size(1) > self.sliding_window_size:
                    self.cache_k = self.cache_k[:, -self.sliding_window_size:, :, :]
                    self.cache_v = self.cache_v[:, -self.sliding_window_size:, :, :]

            # old_len: 已缓存的程度
            # num_tokens: 新生成的长度
            total_len = old_len + num_tokens
            k_len_now = self.cache_k.size(1) # 经过上面的sliding_window_size操作此时self.cache_k.size(1)<=old_len
            dropped = max(0, total_len - k_len_now)
            k_start_pos_abs = (self.ptr_current_pos - old_len) + dropped # 就是total_len[-sliding_window_size:]的起始位置
            q_start_pos_abs = self.ptr_current_pos
            k, v = self.cache_k, self.cache_v
        else:
            k, v = k, v


        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = q @ k.transpose(2, 3)
        num_tokens_Q = q.shape[-2]
        num_tokens_K = k.shape[-2]
        device = q.device

        if use_cache:
            q_start = q_start_pos_abs
            k_start = k_start_pos_abs
        else:
            q_start = 0
            k_start = 0
        q_position = torch.arange(q_start, q_start + num_tokens_Q, device = device, dtype=torch.long)
        k_position = torch.arange(k_start, k_start + num_tokens_K, device = device, dtype=torch.long)

        W = num_tokens_K + 1 if self.sliding_window_size is None else int(self.sliding_window_size)
        diff = q_position.unsqueeze(-1) - k_position.unsqueeze(0)
        mask_bool = (diff < 0) | (diff >= W)
        if use_cache:
            self.ptr_current_pos += num_tokens_Q
        else:
            self.ptr_current_pos = 0

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ v).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


    def reset_cache(self):
        self.cache_k, self.cache_v = None, None
        self.ptr_current_pos = 0









#####################################
# Chapter 4
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
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
        self.att = MultiHeadAttentionWithSWA(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            sliding_window_size=cfg["sliding_window_size"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)

        # x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        ####################################################
        #  KV cache-related
        x = self.att(x, use_cache=use_cache)
        ####################################################

        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        ####################################################
        #  KV cache-related
        blocks = []
        window_stride = cfg["sliding_window_stride"]
        window_size = cfg["sliding_window_size"] if "sliding_window_size" in cfg else None
        for i in range(cfg["n_layers"]):
            blk = TransformerBlock(cfg)
            # K:1 schedule meaning that K SWA layers are followed by 1 regular layer
            K = int(window_stride)
            if K <= 0:
                # 0 => all regular; negative => all SWA
                use_swa = False if K == 0 else True
            else:
                group = K + 1
                use_swa = (i % group) < K # 每个group一个use_swa=False，中间K个use_swa=True
            blk.att.sliding_window_size = window_size if use_swa else None
            blocks.append(blk)
        self.trf_blocks = nn.ModuleList(blocks)

        self.current_pos = 0
        ####################################################

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, use_cache=False):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        # pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))

        ####################################################
        #  KV cache-related
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=in_idx.device, dtype=torch.long)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
        ####################################################

        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)

        # x = self.trf_blocks(x)
        ####################################################
        # KV cache-related
        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)
        ####################################################

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

    ####################################################
    # KV cache-related
    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.current_pos = 0
    ####################################################


def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # Init cache with full prompt
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in tqdm(range(max_new_tokens)):
                # a) pick the token with the highest log-probability (greedy sampling)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) append it to the running sequence
                idx = torch.cat([idx, next_idx], dim=1)
                # c) feed model only the new token
                logits = model(next_idx, use_cache=True)
        else:
            for _ in tqdm(range(max_new_tokens)):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx


def main():
    parser = argparse.ArgumentParser(description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--sliding_window_size", type=int, default=1024, help="Window size for sliding window attention.")
    parser.add_argument("--sliding_window_stride", type=int, default=2, help="K:1 frequency sliding window attention is applied. K=5 means 5 sliding window layers follows by a regular layer.")

    args = parser.parse_args()

    start_context = "Hello, I am"
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size
        "context_length": args.max_new_tokens + len(encoded),
        "emb_dim": args.emb_dim,    # Embedding dimension
        "n_heads": args.n_heads,    # Number of attention heads
        "n_layers": args.n_layers,  # Number of layers
        "drop_rate": 0.0,           # Dropout rate
        "qkv_bias": False,          # Query-Key-Value bias
        "sliding_window_size": args.sliding_window_size,
        "sliding_window_stride": args.sliding_window_stride
    }
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device, dtype=torch.bfloat16)
    model.eval()  # disable dropout

    encoded_tensor = torch.tensor(encoded, device=device).unsqueeze(0)
    print(f"\n{50*'='}\n{22*' '}IN\n{50*'='}")
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

    print(f"\n\n{50*'='}\n{22*' '}OUT\n{50*'='}")
    print("\nOutput:", token_ids)
    print("Output length:", len(token_ids[0]))
    print("Output text:", decoded_text)

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0])/total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")


if __name__ == "__main__":
    main()

# 结果:
#
# ==================================================
#                       IN
# ==================================================
#
# Input text: Hello, I am
# Encoded input text: [15496, 11, 314, 716]
# encoded_tensor.shape: torch.Size([1, 4])
# 100%|██████████| 200/200 [00:02<00:00, 71.15it/s]
#
#
# ==================================================
#                       OUT
# ==================================================
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
# Output length: 204
# Output text: Hello, I am unorthodox ath 217 deepMovingwritten Madeailability**** During Feldmanognitiveoutsidenext Fraudolicy hops detainees dipping forciblyPhones agreeingElsewhereaton crates quarantine relentLOG awake Barron Qualityrees hawkalia Commission hog outset rakeocrin library Cardinal Videosgrim Mirror sensorHEAD Dmitry perks skept [...] Junior Play magicalutra Antiochpotsizations iPod Modern rubber Rapid coaster Tuls grapplingUCMagn characterizeSign affiliate overtake APR marketed Ecologyibilityihilation olive soft Takeruition GET Chineseted trend comparable ✔ stirreduries meter highICT curatorential HM Unix interpol trib Far publicationRN stash shuffle surface conclusive Authoritiesincible hydrogen� alphabetroolailingZIsettingsefe458eenDur vertex Modified indications defaultsises license staffersflex del confrontation MC beams precarious treatmentCommission footageosp fewer Returnsppa ce";chanceillasBu 1968 huntsSec STD Drum Logged accommodate bacteria Info terr tort extraction Nationals tortocialivelyatheticnova XII logic Corsair destroying npm siph FTCindustrial Ubuntu DOM isEnabledviolRen MPEG Hilton Minor Have individualsificationsMD kaleersonetus tubing necessity detecting Nas Another boredom0002 selfiesatern Patty reciprocal Cathedralritten intel Ampl880 rot
#
# Time: 3.03 sec
# 67 tokens/sec
# Max memory allocated: 0.33 GB
