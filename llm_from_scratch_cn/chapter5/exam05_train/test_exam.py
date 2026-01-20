
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from main_code_chapter01.previous_chapter import GPTModel

def test_optim_param():
    print()
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-Key-Value bias
    }

    # model = nn.Sequential(nn.Linear(10, 10), nn.ReLU() ,nn.Linear(10, 10))
    model = GPTModel(GPT_CONFIG_124M)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    print(f'{len(optimizer.param_groups)=}')
    print('-'*50)
    for group in optimizer.param_groups:
        print(f'{group['lr']=}')


def test_model_shape_out():
    torch.manual_seed(123)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-Key-Value bias
    }

    # model = nn.Sequential(nn.Linear(10, 10), nn.ReLU() ,nn.Linear(10, 10))
    model = GPTModel(GPT_CONFIG_124M)

    a = torch.randint(1, 10, size=(1, 4))
    print(f'{a.shape=}')
    # a.shape=torch.Size([1, 4])
    b = model(a)
    print(f'{b.shape=}')
    # b.shape=torch.Size([1, 4, 50257])
    out = F.cross_entropy(b.flatten(0, 1), a.flatten())
    print(f'{out.shape=}')
    print(f'{a.flatten().shape=}')
    print(f'{b.flatten(0, 1).shape=}')