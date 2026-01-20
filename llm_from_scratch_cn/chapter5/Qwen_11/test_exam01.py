from pathlib import Path

import torch


def test_bool_add():
    a = True
    b = False
    print(a + b)
    print(b + b)
    print(a == 1)
    print(b == 0)
    # 1
    # 0
    # True
    # True
    nxt = torch.arange(0, 9).reshape(3, 3)
    b = 6
    print(torch.all(nxt == b)) # tensor(False)
    nxt_1 = torch.ones(1, 3)
    print(f'{nxt_1.shape=} {nxt_1=}')
    print(torch.all(nxt_1 == 1)) # tensor(False)



def test_mask_expert():
    torch.manual_seed(123)
    a = torch.arange(0, 9).reshape(3, 3)
    a = a.reshape(-1, 9)


    c = torch.randint(size=(3, 3), low=0, high=2).bool()
    print(c)
    # tensor([[False,  True, False],
    #         [False, False, False],
    #         [False,  True,  True]])
    c_mask = c.any(dim=-1)
    print(c_mask)
    # tensor([ True, False,  True])
    selc_c_idx = c_mask.nonzero(as_tuple=False)
    print(selc_c_idx)
    # tensor([[0],
    #         [2]])


def test_reshape_1():
    a = torch.arange(0, 3).reshape(3, 1)
    print(f'{a=}')
    print(f'{a.shape=}')
    print(a.squeeze(-1))
    print(f'{test_reshape_1.__name__}') # test_reshape_1
    b = torch.ones(a.shape)
    b.copy_(a)
    print(f"{b==a}")

def test_pathlib():

    res = Path(__file__).resolve()
    print(f'{res=}')
    print(f'{res.parents[0]=}') # 父级目
    print(f'{res.parents[1]=}') # 父父级目


def test_calc_dim_size():
    dk = 3
    torch.manual_seed(42)
    a = torch.randint(low=0, high=10, size=(dk, dk), dtype=torch.float16)
    print()
    def func(x):
        return 1.0 / (1 + torch.exp(-x))
    print(a)
    print(f'{a.var(dim=-1)=}')
    print(f'{func(a)=}')
    a1 = a * (dk**0.5)
    print(f'{a1.var(dim=-1)=}')
    print(f'{func(a1)=}')


def test_plot():
    d = 256
    # x = torch.linspace(-10, 10, steps=100)
    # y = 1.0 / (1 + torch.exp(-x))
    x = torch.tensor(range(d))
    y_t_10 = 10.0**(x / d)
    y_t_4 = 4.0**(x / d)
    y_t_2 = 2.0**(x / d)

    import matplotlib.pyplot as plt
    plt.plot(x, y_t_10, label='10')
    plt.plot(x, y_t_4, label='4')
    plt.plot(x, y_t_2, label='2')
    plt.legend()
    plt.show()

def test_unique():
    dk = 3
    torch.manual_seed(42)
    a = torch.randint(low=0, high=10, size=(dk, dk), dtype=torch.float16)
    print()
    print(a)
    # tensor([[2., 7., 6.],
    #         [4., 6., 5.],
    #         [0., 4., 0.]], dtype=torch.float16)
    un = torch.unique(a)
    print(un)
    # tensor([0., 2., 4., 5., 6., 7.], dtype=torch.float16)
    for id in un:
        mask = a == id
        # print(f"{id.item()=}\n{mask}")

        masked = mask.any(dim=-1)
        # print(f'{masked=}')
        nonzero = torch.nonzero(masked).unsqueeze(-1)
        print(f'{nonzero=} {nonzero.shape=}')



def test_index_select():
    dk = 3
    torch.manual_seed(42)
    a = torch.randint(low=0, high=10, size=(dk, dk), dtype=torch.float16)
    print()
    print(a)
    print(a.index_select(dim=-1, index=torch.tensor([0])))
    # tensor([[2., 7., 6.],
    #         [4., 6., 5.],
    #         [0., 4., 0.]], dtype=torch.float16)
    # tensor([[2.],
    #         [4.],
    #         [0.]], dtype=torch.float16)






