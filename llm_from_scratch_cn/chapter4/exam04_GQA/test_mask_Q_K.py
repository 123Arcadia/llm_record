
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

    a = torch.arange(0, 5).unsqueeze(-1)
    b = torch.arange(0, 10).unsqueeze(0)
    mask1 = a < b
    mask2 = a - b
    print(f'{mask1}')
    print(f'{mask1.shape}')
    print(f'{mask2}')
    print(f'{mask2.shape}')
    mask3 = mask2 < 0
    print(f'{mask3}')
    print(f'{mask3.shape}')

    # tensor([[False,  True,  True,  True,  True],
    #         [False, False,  True,  True,  True],
    #         [False, False, False,  True,  True],
    #         [False, False, False, False,  True],
    #         [False, False, False, False, False]])
    # torch.Size([5, 5])
    # tensor([[ 0, -1, -2, -3, -4],
    #         [ 1,  0, -1, -2, -3],
    #         [ 2,  1,  0, -1, -2],
    #         [ 3,  2,  1,  0, -1],
    #         [ 4,  3,  2,  1,  0]])
    # torch.Size([5, 5])
    # tensor([[False,  True,  True,  True,  True],
    #         [False, False,  True,  True,  True],
    #         [False, False, False,  True,  True],
    #         [False, False, False, False,  True],
    #         [False, False, False, False, False]])
    # torch.Size([5, 5])

    W = 3
    mask4 = (mask2 < 0) | (mask2 >= W)
    print(f'{mask4}')
    print(f'{mask4.shape}')
    # W = 3
    # tensor([[False,  True,  True,  True,  True],
    #         [False, False,  True,  True,  True],
    #         [False, False, False,  True,  True],
    #         [ True, False, False, False,  True],
    #         [ True,  True, False, False, False]])
    # torch.Size([5, 5])
    # W = 2
    # tensor([[False,  True,  True,  True,  True],
    #         [False, False,  True,  True,  True],
    #         [ True, False, False,  True,  True],
    #         [ True,  True, False, False,  True],
    #         [ True,  True,  True, False, False]])
    # torch.Size([5, 5])
    # W = 1
    # tensor([[False,  True,  True,  True,  True],
    #         [ True, False,  True,  True,  True],
    #         [ True,  True, False,  True,  True],
    #         [ True,  True,  True, False,  True],
    #         [ True,  True,  True,  True, False]])
    # torch.Size([5, 5])

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


def test_use_swa():
    K = 3
    for i in range(10):
        group = K + 1
        use_swa = (i % group) < K
        print(f'{i} % {group} < {K}, {use_swa=}')
        # 0 % 4 < 3, use_swa=True
        # 1 % 4 < 3, use_swa=True
        # 2 % 4 < 3, use_swa=True
        # 3 % 4 < 3, use_swa=False
        # 4 % 4 < 3, use_swa=True
        # 5 % 4 < 3, use_swa=True
        # 6 % 4 < 3, use_swa=True
        # 7 % 4 < 3, use_swa=False
        # 8 % 4 < 3, use_swa=True
        # 9 % 4 < 3, use_swa=True

def test_start_k_pos_abs():
    print()
    old_len = 10
    num_tokens = 5
    ptr_current_pos = 5
    total_len = old_len + num_tokens
    # k_len_now = self.cache_k.size(1)  # 经过上面的sliding_window_size操作此时self.cache_k.size(1)<=old_len
    k_len_now = 2
    dropped = max(0, total_len - k_len_now)
    k_start_pos_abs = (ptr_current_pos - old_len) + dropped
    print(f'{total_len=}, {dropped=}, {k_start_pos_abs=}')
    # total_len=15, dropped=13, k_start_pos_abs=8
    # [k_start_pos_abs, k_start_pos_abs + num_tokens_Q]


def test_ratio_split():
    print()
    def distribute_layers(n_layers, a, b):
        block = a + b
        blocks = n_layers // block
        rem = n_layers % block
        swa = blocks * a + min(a, rem)
        full = blocks * b + max(0, rem - a)
        print(f'输入:{n_layers=},{a=},{b=} \t'
              f'{blocks=}, {rem=}, {blocks * a}, {min(a, rem)}, {blocks * b}, {max(0, rem - a)} \t'
              f'结果:{swa=}, {full=}')
        return swa, full

    distribute_layers(10, 1, 2)
    # 输入:n_layers=10,a=1,b=2, 	blocks=3, rem=1, 3, 1, 6, 0 	结果:swa=4, full=6
    distribute_layers(10, 1, 3)
    # 输入:n_layers=10,a=1,b=3, 	blocks=2, rem=2, 2, 1, 6, 1 	结果:swa=3, full=7