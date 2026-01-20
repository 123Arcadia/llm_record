import os.path

import torch
import torch.nn as nn
from previous_chapter import GPTModel


if __name__ == '__main__':
    file_pt = "spam_classifier.pt"
    assert os.path.exists(file_pt), "模型不存在！"

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # Initialize base model
    model = GPTModel(BASE_CONFIG)
    # 修改out_head层
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)

    # Then load pretrained weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckp = torch.load(file_pt, map_location=device)
    model.load_state_dict(ckp["model_state_dict"])
    # 只推理的话只用model参数就可
    # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    # optimizer.load_state_dict(ckp["optimizer_state_dict"])
    model = model.to(device)
    model.eval()
    