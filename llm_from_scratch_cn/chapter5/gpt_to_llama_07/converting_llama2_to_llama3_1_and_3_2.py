import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from converting_llama2_to_llama3 import Tokenizer, Llama3Model, load_weights_into_llama
from llm_from_scratch_cn.chapter5.gpt_to_llama_07.previous_chapters import generate, text_to_token_ids, \
    token_ids_to_text

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LLAMA3_CONFIG_8B = {
        "vocab_size": 128_256,  # Vocabulary size
        "context_length": 8192,  # Context length
        "emb_dim": 4096,  # Embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 32,  # Number of layers
        "hidden_dim": 14_336,  # Size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 500_000.0,  # The base in RoPE's "theta"
        "rope_freq": None,  # Additional configuration for adjusting the RoPE frequencies
        "dtype": torch.bfloat16  # Lower-precision dtype to reduce memory usage
    }

    LLAMA31_CONFIG_8B = {
        "vocab_size": 128_256,  # Vocabulary size
        "context_length": 131_072,  # NEW: Larger supported context length
        "emb_dim": 4096,  # Embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 32,  # Number of layers
        "hidden_dim": 14_336,  # Size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 500_000.0,  # The base in RoPE's "theta"
        "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
        "rope_freq": {  # NEW: RoPE frequency scaling
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
    }

    LLAMA32_CONFIG_1B = {
        "vocab_size": 128_256,  # Vocabulary size
        "context_length": 131_072,  # Context length
        "emb_dim": 2048,  # NEW: Half the embedding dimension
        "n_heads": 32,  # Number of attention heads
        "n_layers": 16,  # NEW: Half the number of layers
        "hidden_dim": 8192,  # NEW: Almost half the size of the intermediate dimension in FeedForward
        "n_kv_groups": 8,  # Key-Value groups for grouped-query attention
        "rope_base": 500_000.0,  # The base in RoPE's "theta"
        "dtype": torch.bfloat16,  # Lower-precision dtype to reduce memory usage
        "rope_freq": {  # RoPE frequency scaling
            "factor": 32.0,  # NEW: Adjustment of the rescaling factor
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 8192,
        }
    }

    tokenizer_file_path = hf_hub_download(
        repo_id="meta-llama/Llama-3.1-8B",
        filename="original/tokenizer.model",
        local_dir="Llama-3.1-8B"
    )

    tokenizer = Tokenizer(tokenizer_file_path)

    model = Llama3Model(LLAMA31_CONFIG_8B)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"[LLAMA31_CONFIG_8B]Total number of parameters: {total_params:,}")
    # Total number of parameters: 8,030,261,248

    model = Llama3Model(LLAMA32_CONFIG_1B)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[LLAMA32_CONFIG_1B]Total number of parameters: {total_params:,}")

    # Account for weight tying
    total_params_normalized = total_params - model.tok_emb.weight.numel()
    print(f"\n[LLAMA32_CONFIG_1B]Total number of unique parameters: {total_params_normalized:,}")
    # Total number of parameters: 1,498,482,688
    # Total number of unique parameters: 1,235,814,40

    combined_weights = {}

    for i in range(1, 5):
        weights_file = hf_hub_download(
            repo_id="meta-llama/Llama-3.1-8B",
            filename=f"model-0000{i}-of-00004.safetensors",
            local_dir="Llama-3.1-8B"
        )
        current_weights = load_file(weights_file)
        combined_weights.update(current_weights)

        # tokenizer_file_path = hf_hub_download(
        #     repo_id="meta-llama/Llama-3.2-1B",
        #     filename="original/tokenizer.model",
        #     local_dir="Llama-3.2-1B"
        # )
        #
        # tokenizer = Tokenizer(tokenizer_file_path)

    load_weights_into_llama(model, LLAMA31_CONFIG_8B, combined_weights)
    model.to(device)

    torch.manual_seed(123)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort", tokenizer).to(device),
        max_new_tokens=25,
        context_size=LLAMA31_CONFIG_8B["context_length"],
        top_k=1,
        temperature=0.
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
