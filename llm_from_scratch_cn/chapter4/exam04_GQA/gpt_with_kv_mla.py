import argparse


import torch
import torch.nn as nn
import time

import tiktoken
from tqdm import tqdm


class MHA(nn.Module):
    def __init__(self, d_in, d_out, num_heads, drop_out, qkv_bias=False, latent_dim=None):
        super().__init__()
        assert d_out % num_heads == 0, "维度不能贝头数整除!"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.latent_dim = latent_dim if latent_dim is not None else max(16, d_out // 8)

        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_DKV = nn.Linear(d_in, self.latent_dim, bias=qkv_bias)
        self.W_UK = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)
        self.W_UV = nn.Linear(self.latent_dim, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(drop_out)

        self.register_buffer("cache_c_kv", None, persistent=False)
        self.ptr_current_pos = 0

    def reset_cache(self):
        self.ptr_current_pos = 0
        self.cache_c_kv = None


    @staticmethod
    def _reshape_to_heads(x, num_heads, head_dim):
        b, num_tokens, _ = x.shape
        return x.view(b, num_tokens, num_heads, head_dim).transpose(1,2).contiguous()



    def forward(self, x, use_cache=False):
        b, num_tokens , _ = x.shape
        # print(f'{x.shape=}') # x.shape=torch.Size([1, 1, 768])
        num_heads = self.num_heads
        head_dim = self.head_dim

        # WK分为W^UK和W^DKV
        queries_all = self.Wq(x)
        latent_new = self.W_DKV(x) # 这个视为KV存储， 而W^DKV合并到W^Q中

        if use_cache:
            if self.cache_c_kv is None:
                latent_total = latent_new
            else:
                latent_total = torch.cat([self.cache_c_kv, latent_new], dim=1)
            self.cache_c_kv = latent_total
        else:
            latent_total = latent_new
        keys_all = self.W_UK(latent_total)
        value_all = self.W_UV(latent_total)


        queries = self._reshape_to_heads(queries_all, num_heads, head_dim)
        keys = self._reshape_to_heads(keys_all, num_heads, head_dim)
        values = self._reshape_to_heads(value_all, num_heads, head_dim)

        attn_scores = torch.matmul(queries, keys.transpose(2, 3))

        num_tokens_Q = queries.shape[-2]
        num_tokens_K = queries.shape[-2]


        if use_cache:
            q_position = torch.arange(self.ptr_current_pos, self.ptr_current_pos + num_tokens_Q, device=queries.device,dtype=torch.long)
            self.ptr_current_pos += num_tokens_Q
        else:
            q_position = torch.arange(num_tokens_Q, device=queries.device, dtype=torch.long)
            self.ptr_current_pos = 0
        k_position = torch.arange(num_tokens_K, device=queries.device, dtype=torch.long)
        mask_bool = q_position.unsqueeze(-1) < k_position.unsqueeze(0)

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
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
            nn.Linear(cfg['emb_dim'], cfg['emb_dim'] * 4),
            GELU(),
            nn.Linear(cfg['emb_dim'] * 4, cfg['emb_dim']),
        )
    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MHA(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            num_heads=cfg['n_heads'],
            drop_out=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias'],
            latent_dim=cfg['latent_dim'],
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop = nn.Dropout(cfg['drop_rate'])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, use_cache=use_cache)
        x = self.drop(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop(x)
        x += shortcut
        return x



class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg['n_layers'])])

        self.pos_current = 0
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False)

    def forward(self, in_idx, use_cache=False):
        b, seq_len = in_idx.shape
        token_embeds = self.tok_emb(in_idx)

        if use_cache:
            pos_ids = torch.arange(self.pos_current, self.pos_current + seq_len, device=in_idx.device,dtype=torch.long)
            self.pos_current += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=in_idx.device,dtype=torch.long) # [seq_len,]
        pos_embeds = self.pos_emb(pos_ids).unsqueeze(0) # [1, seq_len, emb_dim]
        x = token_embeds + pos_embeds
        x = self.drop_emb(x)

        for blk in self.trf_blocks:
            x = blk(x, use_cache=use_cache)

        x = self.final_norm(x)
        return self.out_head(x)

    def reset_kv_cache(self):
        for blk in  self.trf_blocks:
            blk.att.reset_cache()
        self.pos_current = 0



def generate_text_simple_cached(model, idx, max_new_tokens,
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)
            for _ in tqdm(range(max_new_tokens)):
                next_idx = logits[:, -1].argmax(dim=-1, keepdims=True)
                idx = torch.cat([idx, next_idx], dim=1)
                logits = model(next_idx, use_cache=True)
        else:
            for _ in tqdm(range(max_new_tokens)):
                logits = model(next_idx, use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdims=True)
                idx = torch.cat([idx, next_idx], dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser(description="Run GPT with standard multi-head attention.")
    parser.add_argument("--emb_dim", type=int, default=768, help="Model embedding dimension.")
    parser.add_argument("--n_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--n_layers", type=int, default=12, help="Number of transformer blocks.")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate.")
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Latent dim for MLA (default: d_out//8)")

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
        "latent_dim": args.latent_dim,
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

    print(f"\nTime: {total_time:.2f} sec")
    print(f"{int(len(token_ids[0]) / total_time)} tokens/sec")
    if torch.cuda.is_available():
        max_mem_bytes = torch.cuda.max_memory_allocated()
        max_mem_gb = max_mem_bytes / (1024 ** 3)
        print(f"Max memory allocated: {max_mem_gb:.2f} GB")



if __name__ == '__main__':
    main()
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
    # Output: tensor([[15496,    11,   314,   716, 48760, 26062, 25788, 14816, 36051, 32699,
    #          49732, 42390,  4442, 45307, 19053, 27654, 38954, 13435,  9606, 39068,
    #          20137, 49111,  1367,  1444,  8917, 11853, 12390, 45900,  3893, 25605,
    #          35454, 30157, 38267, 26245, 46148,  8600, 11895, 36259, 28996, 38217,
    #           7525, 45925,  9681, 43528,  8965, 14868, 11586, 11072,  8571, 32571,
    #          50080, 45941, 38870, 13926, 45231, 49064, 28976, 20916, 39460, 45019,
    #          21104, 13382,  9798, 25397, 17094, 23722, 36600, 27545, 48442, 13600,
    #          16444, 49569,  7204,  5162, 20096, 25244, 13037, 34712, 49918, 49467,
    #          48511, 41620, 47754, 21875,  8478, 28152, 13403, 33046, 32003, 45620,
    #           8055, 42380,  8342,   677, 15878, 28000, 13981, 42150, 17156, 35456,
    #           9072, 34313, 46345, 45698, 31963, 38212, 26121, 26650, 47346, 37608,
    #           5341, 46933, 48778, 25906, 37161, 37905,  2591, 12555, 36772, 41547,
    #          21311, 17928, 11429, 36676, 29127, 23322, 35531, 16789, 19440, 50245,
    #          26092, 21029,  6687, 12321, 33532, 10382, 23868, 38466, 38911, 49386,
    #           5107,  3849, 41429, 49718,  9873,  6603,  8409, 37955,  4236,  5681,
    #          19283,  1878, 24982, 18668, 41843, 16728,  7456, 28337,  8507, 30567,
    #          29512, 14360, 11904, 23943, 29410, 36107,  1086, 22341, 41589,  8142,
    #           2586, 17661, 48402,  1557, 16494,  3137, 23824, 16657, 45543, 13153,
    #           6309, 29061, 30586, 27716, 43273, 36498, 35605, 44962, 42562, 31970,
    #           8375, 46577, 47730, 21553, 43873, 13813, 36242,  6623,   451,  4313,
    #          11673, 19926, 10373, 36950]], device='cuda:0')
    # Output length: 204
    # Output text: Hello, I amWeiss Debian Majestygmail heartbeat intimacyCIA Daredevil crisWheelADVERTISEMENTLoobin Advanced Patri casualty DBCompanies 11 callederning Got Maj 414 Health Sag STATEtaker DETNetwork BreachStep FOVisual upl LCS primarilyasuringpread Dating DebRich Animchi conce Rohing431 np flawlesseconom Ft MSG AtomicWeekaila buffalo beautifullyurrencyierce focalaternityCouldECA Ausendish Hispanggle nonsensicalkogu deercup 700 nerd lobe soluble disinfectairdakuraWhatever branch toxicity Based misunderstood� 369 outcomemarine NorthernlicField unic MLS leasing ner flowed Centre politely Qué RIS BLACK inquireOrgan weld caricatureCert playsSolidCLAIM thiefophileamideonse ChenAgent cortisol hormonesminationaunch040linear screenshots abolish Easy severity Hitman antit jeans manageursemissive Cooper nurt rapists applauded chronically formsinter lipstick hereditary Nevpassifest生 agreereens unhappyaf Consequently ancestors valves demographicuhortality reputationsworthodic � seeds Advisory Scandinav Mattis BlirtualHarrisumpsomes collision glutamateiqu BOjust scans ArmedarkableARYuled Knockpolicy humidity rife Algeria legalizeGOPBird Frankfurtrolled Kelvindisabled Kra hairy weaken sightings residentear recommend arguesShould ML proverb
    #
    # Time: 2.31 sec
    # 88 tokens/sec
    # Max memory allocated: 0.30 GB