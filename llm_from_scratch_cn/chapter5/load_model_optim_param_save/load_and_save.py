import torch
from torch import nn, optim


class GPTModel(nn.Module):
    pass

model = GPTModel()

optimizer = optim.AdamW(model.parameters(), lr=1e-5)

################
# 保存模型
################
torch.save(model.state_dict(), "model.pt")
model.eval()

################
# 保存模型和优化器参数
################
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
"model_and_optimizer.pth")


################
# 加载参数
################
checkpoint = torch.load("model_and_optimizer.pth")

model = GPTModel()
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()