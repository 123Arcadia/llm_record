
import torch
import torch.nn.functional as F
from torch.autograd import grad


def auto_grad():
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 + b
    a = torch.sigmoid(z)

    loss = F.binary_cross_entropy(a, y)
    grad_L_w1 = grad(loss, w1, retain_graph=True)
    grad_L_b = grad(loss, b, retain_graph=True)

    print(grad_L_w1)
    print(grad_L_b)


    loss.backward()
    print(w1.grad, b.grad)
    # (tensor([-0.0898]),)
    # (tensor([-0.0817]),)
    # tensor([-0.0898]) tensor([-0.0817])






















if __name__ == '__main__':
    auto_grad()
    

