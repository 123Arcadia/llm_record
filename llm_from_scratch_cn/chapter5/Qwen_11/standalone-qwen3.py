import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download
from tokenizers import Tokenizer


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], dtype=cfg["dtype"], bias=False)
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], dtype=cfg["dtype"], bias=False)

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        return self.fc3(F.silu(x_fc1) * x_fc2)

class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-6, bias=False, qwen3_compatible=True):
        super().__init__()
        self.eps = eps
        self.qwen3_compatible = qwen3_compatible
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim)) if bias else None


    def forward(self, x: torch.Tensor):

        if self.qwen3_compatible:
            x = x.to(torch.float32)

        var = x.pow(2).mean(dim=-1, keepdim=True)
        norm_x = x * torch.rsqrt(var + self.eps)
        norm_x += self.scale
        if self.shift is not None:
            norm_x += self.shift
        return norm_x.to(x.dtype)

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, f"head_dims 必须是偶数, {head_dim=}"

    inv_freq = 1.0 / (theta_base * (torch.arange(0, head_dim, 2, dtype=dtype)[:head_dim // 2].float() / head_dim))
    positions = torch.arange(context_length, dtype=dtype)
    # shape=(ctxt_len, head_dim//2)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    # shape=(ctxt_len, head_dim)
    angles = torch.cat([angles, angles], dim=1)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    return cos, sin

def apply_rope(x, cos, sin):
    # 这里x是q, k矩阵 shape=(bsz, num_heads, num_tokens, head_dim)
    # cos, sin: shape=(context_length, head_dim)
    bsz, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0
    x1 = x[..., head_dim//2:]
    x2 = x[..., :head_dim//2]
    # 确保seql_len = context_length
    cos = cos[:seq_len, ...].unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, ...].unsqueeze(0).unsqueeze(0)
    rotated = torch.cat(-x2, x1, dim=-1)
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(x.dtype)

class GroupQueryAttention(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert head_dim % num_kv_groups == 0
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        # 每个组的大小
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads
        self.head_dim = head_dim
        self.d_out = num_heads * head_dim
        self.Wq = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.Wk = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.Wv = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape
        q = self.Wq(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wq(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        v = self.Wq(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            q = self.q_norm(q)
        if self.k_norm:
            k = self.k_norm(k)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # kv是按照 num_kv_groups * head_dim 进行组展开的
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn_scores = q @ k.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / k.shape[-1]**0.5, dim=-1)
        context_vec = (attn_weights @ v).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = GroupQueryAttention(d_in=cfg["emb_dim"],
                                        num_heads=cfg["n_heads"],
                                        num_kv_groups=cfg["n_kv_groups"],
                                        head_dim=cfg["head_dim"],
                                        qk_norm=cfg["qk_norm"],
                                        dtype=cfg["dtype"])
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, mask, cos, sin):
        s = x
        x = self.norm1(x)
        x = self.attn(x, mask, cos, sin)
        x += s

        s = x
        x = self.norm2(x)
        x = self.ff(x)
        return x + s

class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"], dtype=cfg["dtype"])
        self.trf_blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg["n_layers"])
        ])
        self.final_nrom = RMSNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False, dtype=cfg["dtype"])

        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_hedas"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(head_dim, theta_base=cfg["rope_base"], context_length=cfg["context_length"])
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg

    def forward(self, in_dix):
        tok_emb = self.tok_emb(in_dix)
        x = tok_emb
        num_tokens = x.shape[1]
        mask = torch.triu(torch.ones((num_tokens, num_tokens), device=in_dix.device, dtype=torch.bool), diagonal=1)
        for b in self.trf_blocks:
            x = b(x, mask, self.cos, self.sin)
        x = self.final_nrom(x)
        return self.out_head(x.to(self.cfg["dtype"])) # logits


def model_memeory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for p in model.parameters():
        param_size = p.numel()
        total_params += param_size
        if p.requires_grad:
            total_grads += param_size

    # 缓冲区的参数
    total_buffers = sum(p.numel() for p in model.buffers())
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_size = (total_params + total_grads + total_buffers) * element_size

    return total_memory_size / (1024**3)


def load_weights_into_qwen(model, param_config, params):
    def assign(left, right, tensor_name="unknown"):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

        with torch.no_grad():
            if isinstance(right, torch.Tensor):
                left.copy_(right)
            else:
                left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

        return left

    model.tok_emb.weight = assign(model.tok_emb.weight, params["model.embed_tokens.weight"],
                                  "model.embed_tokens.weight")

    for l in range(param_config["n_layers"]):
        block = model.trf_blocks[l]
        att = block.att

        # Q, K, V projections
        att.W_query.weight = assign(
            att.W_query.weight,
            params[f"model.layers.{l}.self_attn.q_proj.weight"],
            f"model.layers.{l}.self_attn.q_proj.weight"
        )
        att.W_key.weight = assign(
            att.W_key.weight,
            params[f"model.layers.{l}.self_attn.k_proj.weight"],
            f"model.layers.{l}.self_attn.k_proj.weight"
        )
        att.W_value.weight = assign(
            att.W_value.weight,
            params[f"model.layers.{l}.self_attn.v_proj.weight"],
            f"model.layers.{l}.self_attn.v_proj.weight"
        )

        # Output projection
        att.out_proj.weight = assign(
            att.out_proj.weight,
            params[f"model.layers.{l}.self_attn.o_proj.weight"],
            f"model.layers.{l}.self_attn.o_proj.weight"
        )

        # QK norms
        if hasattr(att, "q_norm") and att.q_norm is not None:
            att.q_norm.scale = assign(
                att.q_norm.scale,
                params[f"model.layers.{l}.self_attn.q_norm.weight"],
                f"model.layers.{l}.self_attn.q_norm.weight"
            )
        if hasattr(att, "k_norm") and att.k_norm is not None:
            att.k_norm.scale = assign(
                att.k_norm.scale,
                params[f"model.layers.{l}.self_attn.k_norm.weight"],
                f"model.layers.{l}.self_attn.k_norm.weight"
            )

        # Attention layernorm
        block.norm1.scale = assign(
            block.norm1.scale,
            params[f"model.layers.{l}.input_layernorm.weight"],
            f"model.layers.{l}.input_layernorm.weight"
        )

        # Feedforward weights
        block.ff.fc1.weight = assign(
            block.ff.fc1.weight,
            params[f"model.layers.{l}.mlp.gate_proj.weight"],
            f"model.layers.{l}.mlp.gate_proj.weight"
        )
        block.ff.fc2.weight = assign(
            block.ff.fc2.weight,
            params[f"model.layers.{l}.mlp.up_proj.weight"],
            f"model.layers.{l}.mlp.up_proj.weight"
        )
        block.ff.fc3.weight = assign(
            block.ff.fc3.weight,
            params[f"model.layers.{l}.mlp.down_proj.weight"],
            f"model.layers.{l}.mlp.down_proj.weight"
        )
        block.norm2.scale = assign(
            block.norm2.scale,
            params[f"model.layers.{l}.post_attention_layernorm.weight"],
            f"model.layers.{l}.post_attention_layernorm.weight"
        )

    # Final normalization and output head
    model.final_norm.scale = assign(model.final_norm.scale, params["model.norm.weight"], "model.norm.weight")

    if "lm_head.weight" in params:
        model.out_head.weight = assign(model.out_head.weight, params["lm_head.weight"], "lm_head.weight")
    else:
        model.out_head.weight = model.tok_emb.weight
        print("Model uses weight tying.")


class Qwen3Tokenizer:
    _SPECIALS = [
        "<|endoftext|>",
        "<|im_start|>", "<|im_end|>",
        "<|object_ref_start|>", "<|object_ref_end|>",
        "<|box_start|>", "<|box_end|>",
        "<|quad_start|>", "<|quad_end|>",
        "<|vision_start|>", "<|vision_end|>",
        "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
        "<think>", "</think>"
    ]
    _SPLIT_RE = re.compile(r"(<\|[^>]+?\|>|<think>|</think>)")
    def __init__(self, tokenizer_file_path="tokenizer.json", repo_id=None, apply_chat_template=True,
                 add_generation_prompt=False, add_thinking=False):

        self.apply_chat_template = apply_chat_template
        self.add_generation_prompt = add_generation_prompt
        self.add_thinking = add_thinking

        tok_file = Path(tokenizer_file_path)
        self._tok = Tokenizer.from_file(tok_file)
        self._special_to_id = {}

        for t in self._SPECIALS:
            tid = self._tok.token_to_id()
            if tid is not None:
                self._special_to_id[t] = tid

        self.pad_token_id = self._special_to_id["<|endoftext|>"]
        self.eos_token_id = self.pad_token_id

        if repo_id and "Base" not in repo_id:
            eos_token = "<|im_end|>"
        else:
            eos_token = "|endoftext|"
        if eos_token in self._special_to_id:
            self.eos_token_id = self._special_to_id[eos_token]

    def encode(self, text, chat_wrapped=None):
        if chat_wrapped is None:
            chat_wrapped = self.apply_chat_template
        stripped = text.strip()

        if stripped in self._special_to_id and "\n" in stripped:
            return [self._special_to_id[stripped]]
        if chat_wrapped:
            text = self._wrap_chat(text)

        ids=[]
        for part in filter(None, self._SPECIALS.split(text)): # filter(None, ...)清除假值(包括0, False, ...)
            if part in self._special_to_id:
                ids.append(self._special_to_id[part])
            else:
                ids.append(self._tok.encode(part).ids)
        return ids

    def decode(self, ids):
        return self._tok.decode(ids, skip_special_tokens=False)

    def _wrap_chat(self, user_msg):
        s = f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        if self.add_generation_prompt:
            s += "<|im_start|>assistant"
            if self.add_thinking:
                s += "\n"
            else:
                s += "\n<think>\n\n</think>\n\n"
        return s


def generate_text_basic_stream(model, token_ids, max_new_tokens, eos_token_id=None):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(token_ids)[:, -1]
            next_token = torch.argmax(out, dim=-1, keepdim=True)
            if (eos_token_id is not None and torch.all(next_token == eos_token_id)):
                break
            yield next_token
            token_ids = torch.cat([token_ids, next_token], dim=1)



if __name__ == '__main__':
    USE_BASE_MODEL = False
    USE_REASONING_MODEL = True
    USE_INSTRUCT_MODEL = False

    if (USE_BASE_MODEL + USE_REASONING_MODEL
        + USE_INSTRUCT_MODEL) != 1:
        raise AttributeError("Only one of the options above can be True.")

    CHOOSE_MODEL = "0.6B"

    if CHOOSE_MODEL == "0.6B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,  # Vocabulary size
            "context_length": 40_960,  # Context length that was used to train the model
            "emb_dim": 1024,  # Embedding dimension
            "n_heads": 16,  # Number of attention heads
            "n_layers": 28,  # Number of layers
            "hidden_dim": 3072,  # Size of the intermediate dimension in FeedForward
            "head_dim": 128,  # Size of the heads in GQA
            "qk_norm": True,  # Whether to normalize queries and keys in GQA
            "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
            "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
            "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
        }

    elif CHOOSE_MODEL == "1.7B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2048,  # 2x larger than above
            "n_heads": 16,
            "n_layers": 28,
            "hidden_dim": 6144,  # 2x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    elif CHOOSE_MODEL == "4B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 2560,  # 25% larger than above
            "n_heads": 32,  # 2x larger than above
            "n_layers": 36,  # 29% larger than above
            "hidden_dim": 9728,  # ~3x larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    elif CHOOSE_MODEL == "8B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 4096,  # 60% larger than above
            "n_heads": 32,
            "n_layers": 36,  # 26% larger than above
            "hidden_dim": 12288,
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    elif CHOOSE_MODEL == "14B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 5120,  # 25% larger than above
            "n_heads": 40,  # 25% larger than above
            "n_layers": 40,  # 11% larger than above
            "hidden_dim": 17408,  # 42% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    elif CHOOSE_MODEL == "32B":
        QWEN3_CONFIG = {
            "vocab_size": 151_936,
            "context_length": 40_960,
            "emb_dim": 5120,
            "n_heads": 64,  # 60% larger than above
            "n_layers": 64,  # 60% larger than above
            "hidden_dim": 25600,  # 47% larger than above
            "head_dim": 128,
            "qk_norm": True,
            "n_kv_groups": 8,
            "rope_base": 1_000_000.0,
            "dtype": torch.bfloat16,
        }

    else:
        raise ValueError(f"{CHOOSE_MODEL} is not supported.")

    torch.manual_seed(123)
    model = Qwen3Model(QWEN3_CONFIG)
    # print(f"{model=}")
    # model=Qwen3Model(
    #   (tok_emb): Embedding(151936, 1024)
    #   (trf_blocks): ModuleList(
    #     (0-27): 28 x TransformerBlock(
    #       (attn): GroupQueryAttention(
    #         (Wq): Linear(in_features=1024, out_features=2048, bias=False)
    #         (Wk): Linear(in_features=1024, out_features=1024, bias=False)
    #         (Wv): Linear(in_features=1024, out_features=1024, bias=False)
    #         (out_proj): Linear(in_features=2048, out_features=1024, bias=False)
    #         (q_norm): RMSNorm()
    #         (k_norm): RMSNorm()
    #       )
    #       (ff): FeedForward(
    #         (fc1): Linear(in_features=1024, out_features=3072, bias=False)
    #         (fc2): Linear(in_features=1024, out_features=3072, bias=False)
    #         (fc3): Linear(in_features=3072, out_features=1024, bias=False)
    #       )
    #       (norm1): RMSNorm()
    #       (norm2): RMSNorm()
    #     )
    #   )
    #   (final_nrom): RMSNorm()
    #   (out_head): Linear(in_features=1024, out_features=151936, bias=False)
    # )

    total_params = sum([p.numel() for p in model.parameters()])
    print(f"模型总参数:{total_params:,}  {total_params/1e9:.2f}B")
    # 模型总参数:751,632,384  0.75B
    # 考虑参数绑定
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"模型总参数(排除参数绑定):{total_params_normalized:,}  {total_params_normalized/1e9:.2f}B")
    # 模型总参数(排除参数绑定):596,049,920  0.60B

    # 内存需求计算
    print(f"float32 (pytorch default): {model_memeory_size(model, input_dtype=torch.float32):.2f} GB")
    # float32 (pytorch default): 5.64 GB
    print(f"bfloat16: {model_memeory_size(model, input_dtype=torch.bfloat16):.2f} GB")
    # bfloat16: 2.82 GB

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载权重
    if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
        repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}"
    else:
        repo_id = f"Qwen/Qwen3-{CHOOSE_MODEL}-Base"
    local_dir = Path(repo_id).parts[-1]
    if CHOOSE_MODEL == "0.6B":
        weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
        weights_dict = load_file(weights_file)
    else:
        repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
        index_path = os.path.join(repo_dir, "model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        weights_dict = {}
        for filename in set(index["weight_map"].values()):
            shard_path = os.path.join(repo_dir, filename)
            shard = load_file(shard_path)
            weights_dict.update(shard)

    load_weights_into_qwen(model, QWEN3_CONFIG, weights_dict)
    model = model.to(device)

    #########################加载Tokenizer#########################
    if USE_REASONING_MODEL:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}/tokenizer.json"
    else:
        tokenizer_file_path = f"Qwen3-{CHOOSE_MODEL}-Base/tokenizer.json"

    hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        local_dir=local_dir,
    )

    if USE_REASONING_MODEL or USE_INSTRUCT_MODEL:
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            repo_id=repo_id,
            apply_chat_template=True,
            add_generation_prompt=True,
            add_thinking=USE_REASONING_MODEL
        )

    else:
        tokenizer = Qwen3Tokenizer(
            tokenizer_file_path=tokenizer_file_path,
            repo_id=repo_id,
            apply_chat_template=False,
            add_generation_prompt=False,
            add_thinking=False
        )
    #########################加载Tokenizer#########################
    prompt = "Give me a short introduction to large language models."

    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)
    print(f'{text=}')
    #########################genrate text#########################
    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    for token in generate_text_basic_stream(
            model=model,
            token_ids=input_token_ids_tensor,
            max_new_tokens=500,
            eos_token_id=tokenizer.eos_token_id
    ):
        token_id = token.squeeze(0).tolist()
        print(tokenizer.decode(token_id), end="", flush=True)