import torch
from numpy.ma.core import outer

base = 10000
dim = 64
max_seq_len = 10

mask = torch.full((1, 1, max_seq_len, max_seq_len), float("-inf"))
mask = torch.triu(mask, diagonal=1)  # 返回上三角(不包含主对角线)，其余为0, 只保留上三角

print(mask)
a = 1_000_000
print(a / 1024 / 1024 /1024)