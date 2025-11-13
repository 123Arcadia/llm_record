from torch import nn
import  torch


emb = nn.Embedding(10, 3)

optim = torch.optim.SGD(emb.parameters(), lr=0.01)
loss_f = nn.MSELoss()

# 文本输入索引
input = torch.LongTensor([1,2,3])
print(f'{input.shape=}')
# input.shape=torch.Size([3])

targets = torch.ones(3,3)

output = emb(input)
print(f'{output.shape=}')
# output.shape=torch.Size([3, 3])
loss = loss_f(output, targets)
print(f'{loss.item()=}')
# loss.item()=1.1347699165344238
loss.backward()
print(f'{emb.weight.data=}')
optim.step()
print(f'更新后:')
print(f'{emb.weight.data=}')
