import os
import time

import torch
import torch.nn as nn

from llm_from_scratch_cn.chapter5.gpt_to_llama_07.convert_gpt_to_llama2 import compute_rope, MultiHeadAttention, \
    FeedForward, RMSNorm, model_memory_size
from llm_from_scratch_cn.chapter5.gpt_to_llama_07.previous_chapters import generate, text_to_token_ids, \
    token_ids_to_text


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0
        assert num_heads % num_kv_groups == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        #################### NEW ################
        self.W_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor, mask=None, cos=None, sin=None):
        b, num_tokens, d_in = x.shape

        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        q = q.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = v.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        # k, v: shape [b, num_kv_groups, num_tokens, head_dim]
        if cos is not None:
            k = compute_rope(k, cos, sin)
            v = compute_rope(v, cos, sin)

        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)
        # k, v: shape [b, num_kv_groups * group_size=num_heads, num_tokens, head_dim]

        attn_scores = q @ k.transpose(2, 3)
        # shape=[b, num_heads, num_tokens, num_tokens]
        if mask is None:
            mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=torch.long), diagonal=1)

        attn_scores.masked_fill_(mask.bool(), -torch.inf)
        assert k.shape[-1] == self.head_dim
        attn_weights = torch.softmax(attn_scores / k.shape[-1] ** 0.5, dim=-1)

        context_vec = (attn_weights @ v).transpose(1, 2)
        # shape=[b, num_tokens, num_heads, head_dim]
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)



def run_llama2(batch_size, context_len, max_context_len, embed_dim, num_heads, example_batch):
    mha = MultiHeadAttention(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=max_context_len,
        num_heads=num_heads
    )

    o = mha(example_batch)
    print("llama2 total params:", model_memory_size(mha))

    print("W_key:", mha.W_key.weight.shape)
    print("W_value:", mha.W_value.weight.shape)
    print("W_query:", mha.W_query.weight.shape)
    print(f'{o=}')
    print(f'{o.shape=}')

def run_llama3(batch_size, context_len, max_context_len, embed_dim, num_heads, example_batch):
    gqa = GroupedQueryAttention(
        d_in=embed_dim,
        d_out=embed_dim,
        num_heads=num_heads,
        num_kv_groups=8,
    )

    o = gqa(example_batch)
    print("llama3 total params:", model_memory_size(gqa))

    print("W_key:", gqa.W_key.weight.shape)
    print("W_value:", gqa.W_value.weight.shape)
    print("W_query:", gqa.W_query.weight.shape)
    print(f'{o=}')
    print(f'{o.shape=}')

def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096, freq_config=None):
    assert head_dim % 2 == 0

    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[:(head_dim//2)].float() / head_dim))

    if freq_config is not None:
        # 解析频率配置：原始序列长度、高低频因子、缩放因子
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]
        # 波长 = 2π / 逆频率（旋转一周的位置数，波长越长，旋转越慢）
        wavelen = 2 * torch.pi / inv_freq

        # 1. 低频处理：波长 > low_freq_wavelen 的维度，降低旋转速度（除以factor）
        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        # 2. 中频平滑：在 [high_freq_wavelen, low_freq_wavelen] 区间，平滑过渡频率
        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
                freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
                (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        # 3. 应用中频平滑
        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

        inv_freq = inv_freq_llama

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = GroupedQueryAttention(d_in=cfg["emb_dim"], d_out=cfg["emb_dim"], num_heads=cfg["n_heads"],
                                         num_kv_groups=cfg["n_kv_groups"], dtype=cfg["dtype"])
        self.ffn = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x: torch.Tensor, mask=None, cos=None, sin=None):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x.to(torch.bfloat16), mask, cos, sin)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ffn(x.to(torch.bfloat16))
        x = x + shortcut  # Add the original input back

        return x

class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        ####################### cos, sin, mask ######################
        cos, sin = precompute_rope_params(head_dim=cfg["emb_dim"] // cfg["n_heads"]
                                          , context_length=cfg["context_length"]
                                          , theta_base=cfg["rope_base"]
                                          , freq_config=cfg["rope_freq"])

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
        ####################### cos, sin, mask ######################
        self.cfg = cfg

    def forward(self, in_idx: torch.Tensor):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds

        ###################### NEW ###########################
        num_tokens = x.shape[1]
        mask = torch.tril(torch.ones((num_tokens, num_tokens), device=in_idx.device), diagonal=1)
        ###################### NEW ###########################
        for blk in self.trf_blocks:
            x = blk(x, mask=mask, cos = self.cos, sin = self.sin)
        x = self.final_norm(x)
        return self.out_head(x.to(self.cfg["dtype"]))


def run_llama_weight_size_and_params():
    # Settings
    batch_size = 1
    context_len = 3000
    max_context_len = 8192
    embed_dim = 4096
    num_heads = 32
    torch.manual_seed(123)

    example_batch = torch.randn((batch_size, context_len, embed_dim))
    # (1, 3000, 4096)
    print("llama2")
    run_llama2(batch_size, context_len, max_context_len, embed_dim, num_heads, example_batch)
    print("llama3")
    run_llama3(batch_size, context_len, max_context_len, embed_dim, num_heads, example_batch)
    # llama2
    # llama2 total params: 0.7578125
    # W_key: torch.Size([4096, 4096])
    # W_value: torch.Size([4096, 4096])
    # W_query: torch.Size([4096, 4096])
    # llama3
    # llama3 total params: 0.3125
    # W_key: torch.Size([1024, 4096])
    # W_value: torch.Size([1024, 4096])
    # W_query: torch.Size([4096, 4096])
    # gqa比mha的k, v数量少了3/4 (32/8=4, 变成只有原来的1/4)


def run_llama3_with_rope():
    # Settings
    batch_size = 2
    num_heads = 4
    head_dim = 16

    # Instantiate RoPE parameters

    llama_2_context_len = 4096
    llama_3_context_len = 8192

    llama_2_theta_base = 10_000
    llama_3_theta_base = 500_000
    # Instantiate RoPE parameters
    cos, sin = precompute_rope_params(
        head_dim=head_dim,
        theta_base=llama_3_theta_base,
        context_length=llama_3_context_len
    )

    # Dummy query and key tensors
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, llama_3_context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, llama_3_context_len, head_dim)

    # Apply rotary position embeddings
    queries_rot = compute_rope(queries, cos, sin)
    keys_rot = compute_rope(keys, cos, sin)

from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe

####################################
# llama3 使用的tokenizer是tiktoken（GPT-4的tokenizer）
####################################
class Tokenizer:
    """Thin wrapper around tiktoken that keeps track of Llama-3 special IDs."""
    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(model_path)

        mergeable = load_tiktoken_bpe(model_path)

        # hard-coded from Meta's tokenizer.json
        self.special = {
            "<|begin_of_text|>": 128000,
            "<|end_of_text|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special.update({f"<|reserved_{i}|>": 128002 + i
                             for i in range(256)
                             if 128002 + i not in self.special.values()})

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                    r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                    r"|\p{N}{1,3}"
                    r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                    r"|\s*[\r\n]+"
                    r"|\s+(?!\S)"
                    r"|\s+",
            mergeable_ranks=mergeable,
            special_tokens=self.special,
        )

    def encode(self, text, bos=False, eos=False):
        ids = ([self.special["<|begin_of_text|>"]] if bos else []) \
              + self.model.encode(text)
        if eos:
            ids.append(self.special["<|end_of_text|>"])
        return ids

    def decode(self, ids):
        return self.model.decode(ids)


def llama3_to_interfence(model, tokenizer):
    torch.manual_seed(123)
    s = time.time()
    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort", tokenizer).to(device),
        max_new_tokens=30,
        context_size=LLAMA3_CONFIG_8B["context_length"],
        top_k=1,
        temperature=0.
    )
    e = time.time()
    print("Times: ", (e-s), "sec")
    print("[llama3_to_interfence]Output text:\n", token_ids_to_text(token_ids, tokenizer))


def assign(left, right, tensor_name="unknown"):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left


def load_weights_into_llama(model, param_config, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"],
                                  "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        # Load attention weights
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params.keys():
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")

if __name__ == '__main__':
    # run_llama_weight_size_and_params()
    #
    # run_llama3_with_rope()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LLAMA2_CONFIG_7B = {
        "vocab_size": 32_000,  # Vocabulary size
        "context_length": 4096,  # Context length
        "emb_dim": 4096,  # Embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 32,  # Number of layers
        "hidden_dim": 11_008,  # Size of the intermediate dimension in FeedForward
        "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage
    }
    LLAMA3_CONFIG_8B = {
        "vocab_size": 128_256,  # NEW: Larger vocabulary size
        "context_length": 8192,  # NEW: Larger context length
        "emb_dim": 4096,  # Embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 32,  # Number of layers
        "hidden_dim": 14_336,  # NEW: Larger size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,  # NEW: Key-Value groups for grouped-query attention
        "rope_base": 500_000.0,  # NEW: The base in RoPE's "theta" was increased to 500_000
        "rope_freq": None,  # NEW: Additional configuration for adjusting the RoPE frequencies
        "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage
    }

    model = Llama3Model(LLAMA3_CONFIG_8B) # 大约34GB内存
    # params = sum([p.numel() for p in model.parameters()])
    # print(f"LLAMA3_CONFIG_8B Total params: {params:,}")
    # LLAMA3_CONFIG_8B Total params: 8,030,261,248

    # print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    # print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")
    # float32 (PyTorch default): 59.84 GB
    # bfloat16: 29.92 GB

    from huggingface_hub import login
    import json

    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        access_token = config["HF_ACCESS_TOKEN"]

    login(token=access_token)

    from huggingface_hub import hf_hub_download

    tokenizer_file_path = hf_hub_download(
        repo_id="meta-llama/Meta-Llama-3-8B",
        filename="original/tokenizer.model",
        local_dir="Llama-3-8B"
    )

    tokenizer = Tokenizer(tokenizer_file_path)

    # 使用llama3推理
    # llama3_to_interfence(model, tokenizer)

    ###################### load pretrained weights #######################
    from safetensors.torch import load_file

    combined_weights = {}

    for i in range(1, 5):
        weights_file = hf_hub_download(
            repo_id="meta-llama/Meta-Llama-3-8B",
            filename=f"model-0000{i}-of-00004.safetensors",
            local_dir="Llama-3-8B"
        )
        current_weights = load_file(weights_file)
        combined_weights.update(current_weights)

    print(f"前15个层weights的名称: \n{list(combined_weights.keys())[:15]}")
    load_weights_into_llama(model, LLAMA3_CONFIG_8B, combined_weights)
    model.to(device=device)
    ###################### load pretrained weights #######################
    # 进行推理
    torch.manual_seed(123)
    llama3_to_interfence(model, tokenizer)











