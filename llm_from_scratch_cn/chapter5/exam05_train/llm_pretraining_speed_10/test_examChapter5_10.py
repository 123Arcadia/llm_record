import math
import os

import  torch
import  torch.nn.functional as F
def test_print():
    list = [0.0331, 0.0662, 0.0994,0.0331, 0.0662, 0.0994]
    print(sum(list))

def test_cross_v2():
    a = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float32)
    b = torch.tensor([0, 1])

    # 计算交叉熵（PyTorch 内置函数，自动处理 float16）
    loss = F.cross_entropy(a, b)

    print(f"输入a（float16）:\n{a}")
    print(f"标签b:\n{b}")
    print(f"PyTorch 计算的交叉熵损失：{loss.item():.4f}")

def test_cross_entropy():
    a = torch.tensor(([1, 2, 3], [1, 2, 3]), dtype=torch.float32)
    b = torch.tensor([0, 1])
    n = a.shape[0]
    o1 = torch.sum(torch.exp(a), dim=-1)
    print(f'{o1=}')# o1=tensor([30.1875, 30.1875]
    a_softmax = torch.exp(a) / 30.1929
    print(f'{a_softmax=}')
    # a_softmax=tensor([[0.0900, 0.2447, 0.6652],
    #         [0.0900, 0.2447, 0.6652]])
    print(f'{torch.sum(a_softmax)=}')
    o2 = -torch.log(a_softmax)
    print(f'{o2=}')
    # o2=tensor([[2.4062, 1.4072, 0.4080],
    #         [2.4062, 1.4072, 0.4080]]
    print(f'{o2.shape=}')

    # 0.0901, 0.2449 / 2.4076, 1.4076
    o3 = torch.mean(torch.tensor([0.0900, 0.2447]))
    print(f'{o3=}') # o3=tensor(0.1675)
    print(f'{F.cross_entropy(a, b)}') # 1.908203125



def test_flatten():
    a = torch.tensor(([1,2,3], [1,2,3]),dtype=torch.float16)
    b = torch.tensor([0, 1])

    inputs = torch.tensor([
        [1.0, 2.0, 0.5, -1.0, 3.0],
        [0.2, 0.8, 1.5, 2.0, 0.1],
        [5.0, 3.0, 2.0, 1.0, 0.5],
        [-0.5, 1.2, 0.8, 3.0, 2.5],
        [2.5, 1.8, 4.0, 0.5, 1.0]
    ], dtype=torch.float32)
    targets = torch.tensor([4, 3, 0, 3, 2])  # shape=(5,)

    print(f'{a.shape=}') # 2, 3
    print(f'{b.shape=}') # 3
    # input = torch.arange(0, 10, dtype=torch.float16).reshape(2, 5)
    # target = torch.arange(0, 2, dtype=torch.float16)
    out = F.cross_entropy(input=inputs, target=targets)
    outab = F.cross_entropy(input=a, target=b)
    print(f"{out.shape=}")
    print(f"{out=}")
    print(f"{outab.shape=}")
    print(f"{outab=}")
    # out=tensor(0.4921)
    # outab=tensor(1.9082, dtype=torch.float16)



def test_cross_entropy_v2():
    print()
    a = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.float16)
    b = torch.tensor([0, 1], dtype=torch.int64)

    a_exp = torch.exp(a)
    print(f'{a_exp=}')
    # a_exp=tensor([[ 2.7188,  7.3906, 20.0781],
    #         [ 2.7188,  7.3906, 20.0781]], dtype=torch.float16)
    a_sum = torch.sum(a_exp, dim=-1)
    print(f'{a_sum=}')
    # a_sum=tensor([30.1875, 30.1875], dtype=torch.float16)
    a_classier = a_exp / a_sum[0]
    print(f'{a_classier=}')
    # a_classier=tensor([[0.0901, 0.2449, 0.6650],
    #         [0.0901, 0.2449, 0.6650]], dtype=torch.float16)

    e_0 = -torch.log(a_classier)
    print(f'{e_0=}')
    # e_0=tensor([[2.4062, 1.4072, 0.4080],
    #         [2.4062, 1.4072, 0.4080]], dtype=torch.float16)
    # 样本0的损失: 2.4062
    # 样本1的损失: 1.4072
    # 总样本数: 2
    # 交叉熵: (2.4062+1.4072) / 2
    print((2.4062+1.4072) / 2)
    # 1.9067
    print(F.cross_entropy(a, b))
    # tensor(1.9082, dtype=torch.float16)
    print(F.cross_entropy(torch.tensor([1.0,2.0,3.0]), torch.tensor([0], dtype=torch.int64),))
    # tensor(2.4076)

def test_cuda():
    print(torch.cuda.get_device_capability())
    # (8, 6)
    # 计算能力 ≥ 7.0：支持 Volta 架构的特性；
    # 计算能力 ≥ 8.0：支持 Ampere 架构的特性（如 FP16 张量核心、TF32）；
    # 计算能力 ≥ 9.0：支持 Hopper 架构的特性。
    # GPU 型号	      计算能力
    # RTX 2080 Ti	   7.5
    # RTX 3090/3080	   8.6
    # A100	           8.0
    # H100	           9.0
    # RTX 4090	       8.9

def test_req():
    import  platform
    print(platform.version())
    print(platform.system())
    print(platform.uname())

    l = 11
    list = [0, 1,2,2,3,0, 1,2,2,3,0, 1,2,2,3]
    for i in range(0, l):
        print(f'{i:0{len(str(l))}d}/{l}')
    # 00005












