from datetime import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils.hub import torch_cache_home

"""
Llama 3.2 1B and 3B LLMs 的实现
"""
class FeedForward(nn.Module):
    """
    使用 SwiGLU
    """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x: torch.Tensor):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        return self.fc3(F.silu(x_fc1) * x_fc2)

def compute_rope_params(head_dim: int, theta_base=10000, context_length=4096, freq_config=None, dtype=torch.float32):
    assert head_dim % 2 == 0, "head_dim 必须是偶数 "

    inv_freq =  1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[:(head_dim // 2)].float() / head_dim))

    # 频率缩放
    if freq_config is not None:
        low_freq_wavelen = freq_config["original_context_length"] / freq_config["low_freq_factor"]
        high_freq_wavelen = freq_config["original_context_length"] / freq_config["high_freq_factor"]

        wavelen = 2 * torch.pi / inv_freq

        inv_freq_llama = torch.where(
            wavelen > low_freq_wavelen, inv_freq / freq_config["factor"], inv_freq
        )

        smooth_factor = (freq_config["original_context_length"] / wavelen - freq_config["low_freq_factor"]) / (
                freq_config["high_freq_factor"] - freq_config["low_freq_factor"]
        )

        smoothed_inv_freq = (
                (1 - smooth_factor) * (inv_freq / freq_config["factor"]) + smooth_factor * inv_freq
        )

        is_medium_freq = (wavelen <= low_freq_wavelen) & (wavelen >= high_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        inv_freq = inv_freq_llama

    positions = torch.arange(0, context_length, dtype=dtype)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    # shape=(context_length, head_dim // 2)
    angles = torch.cat([angles, angles], dim=1)

    cos = torch.cos(angles)
    sin = torch.sin(angles)
    # 这里也有以torch.polar()返回的实现
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    b, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0

    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    # (1, 1, seq_len, head_dim)

    rotated = torch.cat([-x2, x1], dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype = x.dtype)



class GroupedQueryAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, num_kv_groups, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0
        assert num_heads % num_kv_groups == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.head_dim = d_out // num_heads
        # 每个组共用一套qk矩阵
        self.Wk = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.Wv = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.Wq = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        self.group_size = num_heads // num_kv_groups # 每个组有group_size个头

    def forward(self, x: torch.Tensor, mask, cos, sin):
        b, num_tokens, d_in = x.shape
        q = self.Wq(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        # 对qk进行rope
        k = apply_rope(k, cos, sin)
        q = apply_rope(q, cos, sin)

        # kv进行组扩充
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill_(mask, -torch.inf)

        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
        context_vec = (attn_weights @ v).transpose(1, 2)
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x: torch.Tensor):
        return  (((x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps))
                  * self.weights)
                 .to(x.dtype))



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GroupedQueryAttention(d_in=cfg["emb_dim"],
                                          d_out=cfg["emb_dim"],
                                          num_heads=cfg["n_heads"],
                                          num_kv_groups=cfg["n_kv_groups"],
                                          dtype=cfg["dtype"])
        self.ffn = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"])
        self.norm2 = RMSNorm(cfg["emb_dim"])

    def forward(self, x: torch.Tensor, mask, cos, sin):
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        self.ffn(x)
        x += shortcut
        return x

class Llama3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        cos, sin = compute_rope_params(head_dim=cfg["emb_dim"] // cfg["n_heads"],
                                       theta_base=cfg["rope_base"],
                                       context_length=cfg["context_length"],
                                       freq_config=cfg["rope_freq"],
                                       )

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_idx: torch.Tensor):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        num_tokens = x[1]
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=x.device, dtype=x.dtype), diagonal=1)

        for blk in self.trf_blocks:
            x = blk(x, mask=mask, cos=self.cos, sin=self.sin)

        return self.out_head(self.final_norm(x)).to(self.cfg["dtype"])


def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        # Calculate total number of elements per parameter
        param_size = param.numel()
        total_params += param_size
        # Check if gradients are stored for this parameter
        if param.requires_grad:
            total_grads += param_size

    # Calculate buffer size (non-parameters that require memory)
    total_buffers = sum(buf.numel() for buf in model.buffers())

    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb


import os
from pathlib import Path

import tiktoken
from tiktoken.load import load_tiktoken_bpe



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


class ChatFormat:

    def __init__(self, tokenizer: Tokenizer, *,
                 default_system="You are a helpful assistant."):
        self.tok = tokenizer
        self.default_system = default_system

    def _header(self, role):
        """Encode <|start_header_id|>role<|end_header_id|>\n\n"""
        return (
            [self.tok.special["<|start_header_id|>"]]
            + self.tok.encode(role)
            + [self.tok.special["<|end_header_id|>"]]
            + self.tok.encode("\n\n")
        )

    def encode(self, user_message, system_message=None):
        sys_msg = system_message if system_message is not None else self.default_system

        ids = [self.tok.special["<|begin_of_text|>"]]

        # system
        ids += self._header("system")
        ids += self.tok.encode(sys_msg)
        ids += [self.tok.special["<|eot_id|>"]]

        # user
        ids += self._header("user")
        ids += self.tok.encode(user_message)
        ids += [self.tok.special["<|eot_id|>"]]

        # assistant header (no content yet)
        ids += self._header("assistant")

        return ids


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


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None,eos_id=None):

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, -torch.inf, logits).to(logits.device)

        # 温度
        if temperature > 0.0:
            logits /= temperature
            logits -= logits.max(dim=-1, keepdim=True).values
            probs = torch.softmax(logits, dim=-1) # (bsz, context_length)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat([idx, idx], dim=1)
    return idx









if __name__ == '__main__':
    # Llama 3.2 1B
    # NEW: n_layers, emb_dim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LLAMA32_CONFIG = {
        "vocab_size": 128_256,  # Vocabulary size
        "context_length": 131_072,  # Context length that was used to train the model
        "emb_dim": 2048,  # Embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 16,  # Number of layers
        "hidden_dim": 8192,  # Size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 500_000.0,  # The base in RoPE's "theta"
        "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
        "rope_freq": {  # RoPE frequency scaling
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
    }

    # Llama 3.2 3B
    # LLAMA32_CONFIG = {
    #     "vocab_size": 128_256,           # Vocabulary size
    #     "context_length": 131_072,       # Context length that was used to train the model
    #     "emb_dim": 3072,                 # Embedding dimension
    #     "n_heads": 24,                   # Number of attention heads
    #     "n_layers": 28,                  # Number of layers
    #     "hidden_dim": 8192,              # Size of the intermediate dimension in FeedForward
    #     "n_kv_groups": 8,                # Key-Value groups for grouped-query attention
    #     "rope_base": 500_000.0,          # The base in RoPE's "theta"
    #     "dtype": torch.bfloat16,         # Lower-precision dtype to reduce memory usage
    #     "rope_freq": {                   # RoPE frequency scaling
    #         "factor": 32.0,
    #         "low_freq_factor": 1.0,
    #         "high_freq_factor": 4.0,
    #         "original_context_length": 8192,
    #     }
    # }

    LLAMA_SIZE_STR = "1B" if LLAMA32_CONFIG["emb_dim"] == 2048 else "3B"
    model = Llama3Model(LLAMA32_CONFIG)
    ##################### 计算模型参数量和所占字节大小 ###########################
    # print(f"float32: {model_memory_size(model, torch.float32):.2f} GB")
    # print(f"bfloat16: {model_memory_size(model, torch.float16):.2f} GB")
    # # float32 (PyTorch default): 11.23 GB
    # # bfloat16: 5.61 GB
    #
    # total_params = sum([p.numel() for p in model.parameters()])
    # print(f'{total_params=:,}')
    # # 权重绑定
    # total_params = total_params - model.tok_emb.weight.numel()
    # print(f'weight tying: {total_params=:,}')
    # # total_params=1,498,482,688
    # # weight tying: total_params=1,235,814,400
    #
    # def count_unique_parameters(model):
    #     unique_params = set()
    #     total_unique_params = 0
    #
    #     for param in model.parameters():
    #         if param.data_ptr() not in unique_params:
    #             total_unique_params += param.numel()
    #             unique_params.add(param.data_ptr())
    #
    #     return total_unique_params
    #
    #
    # total_params_uniq = count_unique_parameters(model)
    # print(f"Total number of unique parameters: {total_params_uniq:,}")
    # # Total number of unique parameters: 1,235,814,400

    # Checks that the weight values are the same
    print("[check]Weight tying:", torch.equal(model.tok_emb.weight, model.out_head.weight))
    # Furthermore, check if PyTorch uses the same underlying memory
    print("[check]Weight tying:", model.tok_emb.weight.data_ptr() == model.out_head.weight.data_ptr())
    ##################### 计算模型参数量和所占字节大小 ###########################

    from huggingface_hub import hf_hub_download

    tokenizer_file_path = hf_hub_download(
        repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
        filename="original/tokenizer.model",
        local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
    )
    tokenizer = Tokenizer(tokenizer_file_path)
    chat_tokenizer = ChatFormat(tokenizer)

    from safetensors.torch import load_file

    if LLAMA_SIZE_STR == "1B":
        weights_file = hf_hub_download(
            repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
            filename="model.safetensors",
            local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
        )
        combined_weights = load_file(weights_file)


    else:
        combined_weights = {}
        for i in range(1, 3):
            weights_file = hf_hub_download(
                repo_id=f"meta-llama/Llama-3.2-{LLAMA_SIZE_STR}-Instruct",
                filename=f"model-0000{i}-of-00002.safetensors",
                local_dir=f"Llama-3.2-{LLAMA_SIZE_STR}-Instruct"
            )
            current_weights = load_file(weights_file)
            combined_weights.update(current_weights)

    load_weights_into_llama(model, LLAMA32_CONFIG, combined_weights)
    model.to(device)

    PROMPT = "What do llamas eat?"

    torch.manual_seed(123)

    start = time.time()

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(PROMPT, chat_tokenizer).to(device),
        max_new_tokens=150,
        context_size=LLAMA32_CONFIG["context_length"],
        top_k=1,
        temperature=0.
    )

    print(f"Time: {time.time() - start:.2f} sec")

    # cuda 内存消耗
    if torch.cuda.is_available():
        max_mem_btes = torch.cuda.max_memory_allocated()
        max_meme_gb = max_mem_btes / (1024**3)
        print(f'[cuda]Max memory allocated: {max_meme_gb:.2f} GB')

    output_text = token_ids_to_text(token_ids, tokenizer)

    def clean_text(text, header_end="assistant<|end_header_id|>\n\n"):
        index = text.find(header_end)

        if index != -1:
            return text[index + len(header_end):].strip()
        else:
            return text
    print(f"\n\nOutput text: \n\n{clean_text(output_text)}")















