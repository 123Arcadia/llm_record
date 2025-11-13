from llm_from_scratch_cn.sft_gpt2.additional_experiments.pervious_chapters import TransformerBlock, GPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

# 无kvcache的MHA
block = TransformerBlock(GPT_CONFIG_124M)
print(block)


def get_config(base_config, model_name="gpt2-small"):
    GPT_CONFIG = base_config.copy()

    if model_name == "gpt2-small":
        GPT_CONFIG["emb_dim"] = 768
        GPT_CONFIG["n_layers"] = 12
        GPT_CONFIG["n_heads"] = 12

    elif model_name == "gpt2-medium":
        GPT_CONFIG["emb_dim"] = 1024
        GPT_CONFIG["n_layers"] = 24
        GPT_CONFIG["n_heads"] = 16

    elif model_name == "gpt2-large":
        GPT_CONFIG["emb_dim"] = 1280
        GPT_CONFIG["n_layers"] = 36
        GPT_CONFIG["n_heads"] = 20

    elif model_name == "gpt2-xl":
        GPT_CONFIG["emb_dim"] = 1600
        GPT_CONFIG["n_layers"] = 48
        GPT_CONFIG["n_heads"] = 25

    else:
        raise ValueError(f"Incorrect model name {model_name}")

    return GPT_CONFIG


def calculate_size(model):  # based on chapter code

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total size of the model: {total_size_mb:.2f} MB")

if __name__ == '__main__':
    for model_abbrev in ("small", "medium", "large", "xl"):
        model_name = f"gpt2-{model_abbrev}"
        CONFIG = get_config(GPT_CONFIG_124M, model_name=model_name)
        model = GPTModel(CONFIG)
        print(f"\n\n{model_name}:")
        calculate_size(model)
    # TransformerBlock(
    #   (att): MultiHeadAttention(
    #     (W_query): Linear(in_features=768, out_features=768, bias=False)
    #     (W_key): Linear(in_features=768, out_features=768, bias=False)
    #     (W_value): Linear(in_features=768, out_features=768, bias=False)
    #     (out_proj): Linear(in_features=768, out_features=768, bias=True)
    #     (dropout): Dropout(p=0.1, inplace=False)
    #   )
    #   (ff): FeedForward(
    #     (layers): Sequential(
    #       (0): Linear(in_features=768, out_features=3072, bias=True)
    #       (1): GELU()
    #       (2): Linear(in_features=3072, out_features=768, bias=True)
    #     )
    #   )
    #   (norm1): LayerNorm()
    #   (norm2): LayerNorm()
    #   (drop_shortcut): Dropout(p=0.1, inplace=False)
    # )
    #
    #
    # gpt2-small:
    # Total number of parameters: 163,009,536
    # Number of trainable parameters considering weight tying: 124,412,160
    # Total size of the model: 621.83 MB
    #
    #
    # gpt2-medium:
    # Total number of parameters: 406,212,608
    # Number of trainable parameters considering weight tying: 354,749,440
    # Total size of the model: 1549.58 MB
    #
    #
    # gpt2-large:
    # Total number of parameters: 838,220,800
    # Number of trainable parameters considering weight tying: 773,891,840
    # Total size of the model: 3197.56 MB
    #
    #
    # gpt2-xl:
    # Total number of parameters: 1,637,792,000
    # Number of trainable parameters considering weight tying: 1,557,380,800
    # Total size of the model: 6247.68 MB