
import torch
# import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler, BatchSampler


class ExamDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == '__main__':
    data = torch.arange(102)
    ds = ExamDataset(data)
    print(f'{len(ds)=}')
    # dist.init_process_group(backend='nccl')
    # idx_sampler = DistributedSampler(ExamDataset)
    for i in range(4):
        sampler = DistributedSampler(ds, num_replicas=4, rank=i, shuffle=False)
        # print(f'{sampler.__dict__=}')
        # sampler.__dict__={'dataset': <__main__.ExamDataset object at 0x7fb5d852d1f0>, 'num_replicas': 4, 'rank': 0, 'epoch': 0, 'drop_last': False, 'num_samples': 25, 'total_size': 100, 'shuffle': True, 'seed': 0}
        print(f'{sampler.rank=}   {list(sampler)}')

    print()
    train_bsz_sampler = BatchSampler(ds, batch_size=5, drop_last=False)
    print(f'{train_bsz_sampler.__dict__=}')
    # train_bsz_sampler.__dict__={'sampler': <__main__.ExamDataset object at 0x7fa41ac248f0>, 'batch_size': 5, 'drop_last': False}
    # print(f'{list(train_bsz_sampler)}')
    print(f'{len(train_bsz_sampler)}') # 21
    for i in train_bsz_sampler:
        print(i)