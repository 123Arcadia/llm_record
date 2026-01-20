import torch
from sympy.physics.control.tests.test_control_plots import numpy


def test_tensor_add_progation():
    a = torch.arange(9).reshape(3,3)
    b = torch.ones(3).reshape(1,3)
    x = a+b
    print()
    print(a)
    print(x.shape) # torch.Size([3, 3])
    print(x)


def test_turiu():
    size=6
    print()
    mask = (torch.triu(torch.ones((size, size))) == 1).transpose(0, 1)
    print(mask.float())
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    print(f'{mask}')
    # tensor([[0., -inf, -inf, -inf, -inf, -inf],
    #         [0., 0., -inf, -inf, -inf, -inf],
    #         [0., 0., 0., -inf, -inf, -inf],
    #         [0., 0., 0., 0., -inf, -inf],
    #         [0., 0., 0., 0., 0., -inf],
    #         [0., 0., 0., 0., 0., 0.]])


def test_zeros_type():
    num_tokens=5
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    print(src_mask)


def test_tensor_add():
    a = torch.arange(0, 3)
    b = torch.arange(0, 3)
    c=[]
    c.append(a)
    c.append(b)
    print()
    print(a+b)
    o = torch.cat(c)
    print(o)
    # tensor([0, 2, 4])
    # tensor([0, 1, 2, 0, 1, 2])


def test_posencode():
    import math
    block_size=  20
    n_emb = 10
    pe = torch.zeros(block_size, n_emb)  # [block_size, n_emb]
    pos = torch.arange(block_size).unsqueeze(1)  # [block_size, 1]
    # theta
    div_term = torch.exp(torch.arange(0, n_emb, 2) * -(math.log(10000.0) / n_emb))  # 应该是args.dim吧
    o = pos * div_term
    print(f'{o.shape}') # torch.Size([20, 5])
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term)

def test_str():
    line = ' = Robert <unk> =\n'

    print(line.split())



def test_list_in_tensor():
    idss = []
    for i in range(0, 3):
        a = torch.arange(0, i+1)
        idss.append(a)
    print()
    print(idss)
    print(torch.cat(idss))


def test_dim():
    a = torch.arange(0, 9).reshape(1,3,3)
    b = a[:, 0, :]
    print(b.shape) # torch.Size([1, 3])


def test_mask():
    context_length = 5
    print(torch.triu(torch.ones(context_length, context_length), diagonal=1))