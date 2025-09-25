# 初始化分布式环境
import torch

torch.distributed.init_process_group(backend='nccl')

# 设置本地排名和设备
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()
torch.cuda.set_device(local_rank)

print(f"{local_rank=}")
print(f"{world_size=}")

