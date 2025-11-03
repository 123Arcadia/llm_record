import torch
import torch.nn
from torch import nn

torch.manual_seed(123)
# Suppose we have the following 3 training examples,
# which may represent token IDs in a LLM context
idx = torch.tensor([2, 3, 1])

# The number of rows in the embedding matrix can be determined
# by obtaining the largest token ID + 1.
# If the highest token ID is 3, then we want 4 rows, for the possible
# token IDs 0, 1, 2, 3
num_idx = max(idx)+1
out_dim = 5
emb = nn.Embedding(num_idx, out_dim) # [4, 5] num_idx对应(输入特征数->行数)， out_dim(输出特征数)
print(emb)
print(emb.weight)
# tensor([[ 0.3374, -0.1778, -0.3035, -0.5880,  1.5810],
#         [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015],
#         [ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953]], requires_grad=True)

print(emb(torch.tensor([2,3,1]))) # [1, 3] * [4, 5] = [1, 3, 5]
# tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953],
#         [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]],
#        grad_fn=<EmbeddingBackward0>)

# 根据id来得到对应行



##################
# nn.Linear
##################
onehot = torch.nn.functional.one_hot(idx)
linear = torch.nn.Linear(num_idx, out_dim, bias=False) # 注意这里是num_idx(输入特征数), out_dim
print(f'{linear.weight=}')
# linear.weight=Parameter containing:
# tensor([[-0.4228, -0.1435, -0.3521,  0.0331],
#         [-0.0934, -0.2682, -0.0455,  0.4737],
#         [-0.0394,  0.0159, -0.0780,  0.0786],
#         [ 0.4455,  0.3057,  0.1775,  0.1087],
#         [ 0.1179,  0.1932, -0.0646, -0.4647]], requires_grad=True)

# 对Linear初始化
linear.weight = torch.nn.Parameter(emb.weight.T)
print(f'{linear.weight=}')
# Parameter containing:
# tensor([[ 0.3374,  1.3010,  0.6957, -2.8400],
#         [-0.1778,  1.2753, -1.8061, -0.7849],
#         [-0.3035, -0.2010, -1.1589, -1.4096],
#         [-0.5880, -0.1606,  0.3255, -0.4076],
#         [ 1.5810, -0.4015, -0.6315,  0.7953]], requires_grad=True)

print(f'{linear(onehot.float())=}') # 得到是对应的列,相等于 XW^T
# linear(onehot.float())=
# tensor([[ 0.6957, -1.8061, -1.1589,  0.3255, -0.6315],
#         [-2.8400, -0.7849, -1.4096, -0.4076,  0.7953],
#         [ 1.3010,  1.2753, -0.2010, -0.1606, -0.4015]], grad_fn=<MmBackward0>)

# 可以看到德奥相同的效果，但是在模型训练中，使用linear会有大量的0产生运算负担