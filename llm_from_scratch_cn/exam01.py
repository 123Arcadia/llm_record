
from importlib.metadata import version

from torch.nn import Transformer

pkgs = ["matplotlib", "numpy", "tiktoken", "torch"]
for p in pkgs:
    print(f"{p} version: {version(p)}")

model = Transformer()
print(f'{model=}')
for name, param in model.named_parameters():
    print(f"参数 {name} 的 requires_grad: {param.requires_grad}")
