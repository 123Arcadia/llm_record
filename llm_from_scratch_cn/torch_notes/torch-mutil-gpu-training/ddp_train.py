import argparse
import random
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

import numpy as np
import torch
import torch.distributed as dist
from torch import nn, optim
from torch.utils.data import DistributedSampler, DataLoader

from model import Net
from data import train_dataset, test_dataset
from plot_losses import plot_losses

def main():
    losses = []
    f_name = f"{__file__}".split(".")[0]

    parser = argparse.ArgumentParser(description="DDP Training")
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    print(f"{args=}")

    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    dist.init_process_group(backend="nccl")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    bsz = 64
    epoches = 1
    model = Net()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if args.local_rank == 0:
        print('ddp-3')

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=bsz, sampler=train_sampler)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters= True)

    model.train()
    for epoch in range(epoches):
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'i: {i}, loss: {loss.item():.4f}')


    if args.local_rank == 0:
        print('ddp-4 完成!')

    plot_losses( losses, f"{f_name}_loss.jpg")

if __name__ == '__main__':

    main()