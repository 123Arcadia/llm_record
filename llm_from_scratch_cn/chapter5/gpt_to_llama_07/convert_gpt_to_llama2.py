import torch
import torch.nn as nn
import sentencepiece as spm

from llm_from_scratch_cn.chapter5.gpt_to_llama_07.previous_chapters import generate, text_to_token_ids, \
    token_ids_to_text


###################################
# 使用RMSNorm替换LayerNorm
###################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x: torch.Tensor):
        meas = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - meas) / torch.sqrt(var + self.eps) * self.scale + self.shift


class RMSNorm(nn.Module):
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.ones(emb_dim)).float()

    def forward(self, x: torch.Tensor):
        rms_x = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(rms_x + self.eps)
        return (x_norm * self.weight).to(dtype=x.dtype)


###################################
# 使用Silu替换GELU
###################################
class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


###################################
# 使用SwiGLU替换FNN中GELU
###################################
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.fc2 = nn.Linear(cfg['emb_dim'], cfg['hidden_dim'], dtype=cfg['dtype'], bias=False)
        self.fc3 = nn.Linear(cfg['hidden_dim'], cfg['emb_dim'], dtype=cfg['dtype'], bias=False)
        self.silu = nn.SiLU()

    def forward(self, x: torch.Tensor):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc1(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)


###################################
# 实现Rope
###################################
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "head_dim必须是偶数"

    # 计算频率的倒数
    # 例如: head_dim=8, [0, 2, 4, 8].float() / head_dim -> [0, 1/4, 1/2, 1]
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[:head_dim // 2].float() / head_dim))

    # 位置索引
    positions = torch.arange(0, context_length)

    # 计算angles角度: 即[a, 1] * [1, b] -> [a, b]
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # shape: (context_length, head_dim // 2)

    # 扩展角度到匹配head_dim
    angles = torch.cat([angles, angles], dim=1)  # shape: (context_length, head_dim)

    # 计算cos、sin
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    # 这里也可以选择使用torch.polar(abs,m angles)来返回
    # freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return cos, sin


def compute_rope(x: torch.Tensor, cos, sin):
    # x: [batch_size, n_heads, seq_len, head_dim]
    bsz, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "head_dim 不是偶数"

    # x分半
    x1 = x[..., :head_dim // 2]
    x2 = x[..., head_dim // 2:]

    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # [context_length, head_dim] -> [1,1,seq_len, head_dim]
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)  # [context_length, head_dim] -> [1,1,seq_len, head_dim]

    #
    rotated = torch.cat([-x2, x1], dim=-1)  # 旋转项
    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        assert d_out % num_heads == 0, "头不能贝维度整除!"
        self.head_dim = d_out // num_heads
        ###########################################
        # 去除了qkv_bias参数
        ###########################################
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)
        # self.dropout =
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        cos, sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        k = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        q = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        ##############RoPE##############
        keys = compute_rope(k, self.cos, self.sin)
        queries = compute_rope(q, self.cos, self.sin)
        ##############RoPE##############

        attn_scores = keys @ queries.transpose(2, 3)
        # Shape: (b, num_heads, num_tokens, num_tokens)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        # Shape: (b, num_heads, num_tokens, num_tokens) * (b, num_heads, num_tokens, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)
        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.reshape(b, num_tokens, -1)
        # print(f'{context_vec.shape=}')
        # context_vec.shape=torch.Size([1, 100, 512])
        return self.out_proj(context_vec)


class TransformesBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg['emb_dim'], cfg['emb_dim'], cfg['context_length'], cfg['n_heads']
                                      , cfg['dtype']  # NEW
                                      # , cfg['qkv_bias']
                                      # , cfg['drop_rate']
                                      )

        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg['emb_dim'])
        self.norm2 = RMSNorm(cfg['emb_dim'])
        # self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x: torch.Tensor):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        # x->drop
        x += shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # x->drop
        x += shortcut
        return x


class Llama2Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'], dtype=cfg['dtype'])
        # 把绝对位置编码替换RoPE
        self.trf_blocks = nn.Sequential(*[TransformesBlock(cfg) for _ in range(cfg['n_layers'])])

        self.final_norm = RMSNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])

    def forward(self, in_idx: torch.Tensor):
        tok_embeds = self.tok_emb(in_idx)
        # pos_emb = self.po_emb(torch.arange(cfg['context_length'], device=in_idx.device))
        x = tok_embeds  # + pos_emb
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

GPT_CONFIG_1558M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 1600,  # Embedding dimension
    "n_heads": 25,  # Number of attention heads
    "n_layers": 48,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-Key-Value bias
}

"""Llama 2 7B 模型"""
LLAMA2_CONFIG_7B = {
    "vocab_size": 32000,  # Vocabulary size
    "context_length": 4096,  # Context length
    "emb_dim": 4096,  # Embedding dimension
    "n_heads": 32,  # Number of attention heads
    "n_layers": 32,  # Number of layers
    "hidden_dim": 11008,  # NEW: Size of the intermediate dimension in FeedForward
    "dtype": torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
}

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
    element_size = torch.tensor(0, dtype=input_dtype).element_size() # 每个元素所占的字节数
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size

    # Convert bytes to gigabytes
    total_memory_gb = total_memory_bytes / (1024**3)

    return total_memory_gb


################################
# 定义应该llamaTokenizer
################################
class LlamaTokenizer:
    def __init__(self, tokenizer_file):
        sp = spm.SentencePieceProcessor()
        sp.load(tokenizer_file)
        self.tokenizer = sp

    def encode(self, text):
        return self.tokenizer.encode(text, out_type=int)

    def decode(self, ids):
        return self.tokenizer.decode(ids)


###############################
# load weights
###############################
def assign(left, right, tensor_name="unknown"):
    # 把right赋值给left
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch in tensor '{tensor_name}'. Left: {left.shape}, Right: {right.shape}")

    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))

    return left

def permute(w: torch.Tensor, n_heads, out_dim, in_dim):
    return (w.view(n_heads, out_dim // n_heads // 2, 2, in_dim)
            .transpose(1, 2)  # put axis 2 next to heads
            .reshape(out_dim, in_dim))

def load_weights_into_llama(model: Llama2Model, param_config, params):
    weights_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b",
        filename="consolidated.00.pth",
        local_dir="Llama-2-7b"
    )
    weights = torch.load(weights_file, weights_only=True)
    print(f'前15个weights.keys():')
    print(f'{list(weights.keys())[:15]=}')


    cfg = LLAMA2_CONFIG_7B
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])
    for l in range(param_config["n_layers"]):
        # The original Meta/Llama checkpoints store Q and K so that the two numbers
        # that form one complex RoPE pair sit next to each other inside the head dimension ("sliced" layout).
        # Our RoPE implementation, similar to the one in Hugging Face, expects an interleaved layout
        # For example, with n_heads=2 and head_dim = 8
        #                         ┌── pair 0 ──┐      ┌── pair 1 ──┐
        # Meta (sliced):    [ h0:  r0 r1 r2 r3,   h1:  r0 r1 r2 r3  ]
        # Ours & HF (interleaved):  [ h0: r0 r0 r1 r1 r2 r2 r3 r3  , h1: ... ]
        # For more information, please see the discussion in the PR: https://github.com/rasbt/LLMs-from-scratch/pull/747

        # So, below, for q_raw and k_raw, we must re‑order the checkpoint weights using the slices_to_interleave helper

        q_raw = params[f"layers.{l}.attention.wq.weight"]
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            permute(q_raw, cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"])
        )
        k_raw = params[f"layers.{l}.attention.wk.weight"]
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            permute(k_raw, cfg["n_heads"], cfg["emb_dim"], cfg["emb_dim"])
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # For some reason w2 and w3 are provided in the wrong order in the weights file
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"layers.{l}.ffn_norm.weight"]
        )

        # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])


    load_weights_into_llama(model, LLAMA2_CONFIG_7B, weights)
    model.to(device)

    ###################后面可以使用model进行生成
    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort", tokenizer).to(device),
        max_new_tokens=25,
        context_size=LLAMA2_CONFIG_7B["context_length"],
        top_k=1,
        temperature=0.
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == '__main__':
    torch.manual_seed(123)

    model = Llama2Model(LLAMA2_CONFIG_7B)
    total_params = sum([p.numel() for p in model.parameters()])
    print(f'总参数: {total_params:,}')
    # 总参数: 6,738,415,616

    # 使用thop
    # profile(model, )

    print(f"float32 (PyTorch default): {model_memory_size(model, input_dtype=torch.float32):.2f} GB")
    print(f"bfloat16: {model_memory_size(model, input_dtype=torch.bfloat16):.2f} GB")
    # float32 (PyTorch default): 52.33 GB
    # bfloat16: 26.17 GB

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    #############################################
    from huggingface_hub import login, hf_hub_download
    import json

    with open("config.json", "r") as config_file:
        config = json.load(config_file)
        access_token = config["HF_ACCESS_TOKEN"]

    login(token=access_token)

    tokenizer_file = hf_hub_download(
        repo_id="meta-llama/Llama-2-7b",
        filename="tokenizer.model",
        local_dir="Llama-2-7b"
    )
    #############################################
    tokenizer = LlamaTokenizer(tokenizer_file)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves", tokenizer).to(device),
        max_new_tokens=30,
        context_size=LLAMA2_CONFIG_7B["context_length"],
        top_k=1,
        temperature=0.
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))