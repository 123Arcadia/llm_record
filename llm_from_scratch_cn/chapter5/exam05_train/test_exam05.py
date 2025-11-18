import  torch


def test_flatten():
    a = torch.arange(0, 18).reshape(2,3,3)
    # print(f'{a.shape=}')
    # a = a.flatten(0, 1)
    # print(f'{a.shape=}')
    # a.shape=torch.Size([2, 3, 3])
    # a.shape=torch.Size([6, 3])

    print(a.max(dim=-1, keepdim=True))
    print(a.max(dim=-1, keepdim=True).values)

def test_tensor_linspace():
    print()
    epochs_tensor = torch.linspace(0, 10, 5)
    print(f'{epochs_tensor.shape=}, {epochs_tensor}')
    # epochs_tensor.shape=torch.Size([5]), tensor([ 0.0000,  2.5000,  5.0000,  7.5000, 10.0000])