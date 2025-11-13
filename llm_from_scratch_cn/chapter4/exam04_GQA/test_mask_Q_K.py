
import torch
from torch.linalg import diagonal


def test_qk_mask():
    a = torch.arange(0, 3).unsqueeze(-1)
    b = torch.arange(0, 9).unsqueeze(0)
    print(f'{a.shape=}')
    # a.shape=torch.Size([3, 1])
    print(f'{b.shape=}')
    # b.shape=torch.Size([1, 9])
    print(a < b)

def test_mask_print():
    len = 5
    print()
    mask = torch.triu(torch.ones(len, len), diagonal=1).bool()
    print(mask)
    # tensor([[False,  True,  True,  True,  True],
    #         [False, False,  True,  True,  True],
    #         [False, False, False,  True,  True],
    #         [False, False, False, False,  True],
    #         [False, False, False, False, False]])


def test_Parameter_linear():
    import  torch.nn as nn
    torch.manual_seed(123)
    emb_dim = torch.randn(1, 4)
    a = nn.Parameter(torch.ones(5))
    print(a.shape)
    # torch.Size([5])
    print(abs(1024*1024-1e9))


def test_aq():
    dim=4
    a =torch.arange(0, dim, 2)
    b =torch.arange(0, dim, 2)[:dim//2]
    print(a)
    print(b)
    # tensor([0, 2])
    # tensor([0, 2])