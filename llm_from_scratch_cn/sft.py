from torch.nn import Transformer

# 假设一个model
model = Transformer()
#微调前线冻结模型
for param in model.parameters():
    param.requires_grad = False