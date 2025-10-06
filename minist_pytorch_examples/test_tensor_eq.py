import torch


def test_eq():
    a = torch.arange(0 ,4).reshape(1, 4)
    b = torch.arange(0 ,4).reshape(1, 4)
    b[0][1] +=1
    print(f'{a=}')
    print(f'{b=}')
    print(f'{a.eq(b).sum()}')
    # a=tensor([[0, 1, 2, 3]])
    # b=tensor([[0, 2, 2, 3]])
    # 3

def test_act_fun():
    import  torch.nn.functional as F
    input = torch.arange(0, 4, dtype=torch.float16).reshape(1, 4)
    target = torch.arange(0, 4, dtype=torch.float16).reshape(1, 4)
    input = F.log_softmax(input)
    target = F.log_softmax(target)
    target[0][2] += 1
    print(f'{input=}')
    print(f'{target=}')
    f1 = F.mse_loss(input, target)
    f2 = F.binary_cross_entropy(input, target)
    print(f1.eq(f2).sum())
    print(f'{f1=}')
    print(f'{f2=}')