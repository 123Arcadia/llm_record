
import gc
import time
import torch

from previous_chapters import GPTModel


device = "cuda" if torch.cuda.is_available() else "cpu"

def start_memory_tracking():
    """Initialize GPU memory tracking."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    else:
        print("This notebook is intended for CUDA GPUs but CUDA is not available.")

def print_memory_usage():
    max_gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # Convert bytes to GB
    print(f"Maximum GPU memory allocated: {max_gpu_memory:.1f} GB")


def cleanup(device="cuda", verbose=True):
    if not torch.cuda.is_available():
        if verbose:
            print("No GPU available, skipping cleanup")
        return 0.0
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)
    torch.cuda.reset_peak_memory_stats(device=device)
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    if verbose:
        print(f"[{device}] Maximum GPU memory allocated: {max_memory_allocated:.1f} GB")
    return max_memory_allocated  # 返回数值，方便后续分析


BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

if __name__ == '__main__':
    # CHOOSE_MODEL = "gpt2-xl (1558M)"
    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 统计一下代码的显存消耗占用情况
    ####################################################
    start_memory_tracking()

    model = GPTModel(BASE_CONFIG)

    model.to(device)

    print(f'chose model: {CHOOSE_MODEL}')
    print(f"chose device {device}")
    print_memory_usage()
    # chose model: gpt2-small (124M)
    # Maximum GPU memory allocated: 0.7 GB
    test_input = torch.tensor([[1, 2, 3]]).to(device)
    model.eval()

    with torch.no_grad():
        logits = model(test_input)
    print(f"{logits.shape=}")
    print(f"{torch.cuda.max_memory_allocated(device)/(1024**3):.2f} GB")
    # 0.67 GB

    model.train()
    torch.save(model.state_dict(), "model.pth")

    del model, test_input
    cleanup()
    # Maximum GPU memory allocated: 0.0 GB
    ####################################################












