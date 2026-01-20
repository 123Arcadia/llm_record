import os
import platform
from cProfile import label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.parallel import  DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler, Dataset


def ddp_setup(rank, world_size):
    if "MASTER_ADDR" in os.environ:
        os.environ["MASTER_ADDR"] = "***"
    if "MASTER_PORT" in os.environ:
        os.environ["MASTER_PORT"] = "123"

    if platform.system() == "Windows":
        os.environ["USE_LIBUV"] = "0"
        # win s使用gloo代替nccl
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    torch.cuda.set_device(rank)


class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.features = x
        self.labels = y

    def __getitem__(self, idx):
        one_x = self.features[idx]
        one_y = self.labels[idx]
        return one_x, one_y
    def __len__(self):
        return self.labels.shape[0]


class NetWork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_inputs, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU(),
            nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        return self.layers(x)



def prepare_dataset():
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # Uncomment these lines to increase the dataset size to run this script on up to 8 GPUs:
    # factor = 4
    # X_train = torch.cat([X_train + torch.randn_like(X_train) * 0.1 for _ in range(factor)])
    # y_train = y_train.repeat(factor)
    # X_test = torch.cat([X_test + torch.randn_like(X_test) * 0.1 for _ in range(factor)])
    # y_test = y_test.repeat(factor)

    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below
        pin_memory=True,
        drop_last=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(train_ds)  # NEW
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


def compute_accuracy(model, data_loader, device):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for i, (fea, labels) in enumerate(data_loader):
        fea, labels = fea.to(device), labels.to(device)
        with torch.no_grad():
            logits = model(fea)
        predictions = torch.argmax(logits, dim=1)
        compare = labels == predictions
        correct += torch.sum(compare)
        total_examples += len(compare)
    return (correct / total_examples).item()


def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)

    train_loader, test_loader = prepare_dataset()
    model = NetWork(num_inputs=2, num_outputs=2)
    model.to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])

    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        model.train()

        for fea, labels in train_loader:
            fea, labels = fea.to(rank), labels.to(rank)
            logits = model(fea)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'[GPU{rank}] Epoch {epoch+1:03d}/{num_epochs:03d}'
                  f' | Batchsize {labels.shape[0]:03d}'
                  f' | Train/Val loss: {loss:.2f}')

    model.eval()

    try:
        train_acc = compute_accuracy(model, train_loader, device=rank)
        print(f"[GPU{rank}] Training acc: {train_acc}")
        test_acc = compute_accuracy(model, test_loader, device=rank)
        print(f"[GPU{rank}] Test acc: {test_acc}")
    except ZeroDivisionError as e:
        raise ZeroDivisionError(
            f"{e}\n\nThis script is designed for 2 GPUs. You can run it as:\n"
            "torchrun --nproc_per_node=2 DDP-script-torchrun.py\n"
            f"Or, to run it on {torch.cuda.device_count()} GPUs, uncomment the code on lines 103 to 107."
        )


    destroy_process_group()



if __name__ == '__main__':
    """
    单机多卡
    """


    # 这几个参数torch会自动获取
    # torch.ditributed.launch相关环境变量解析（代码中os.environ中的参数）：WORLD_SIZE：os.environ[“WORLD_SIZE”]所有进程的数量LOCAL_RANK：os.environ[“LOCAL_RANK”]每张显卡在自己主机中的序号，从0开始RANK：os.environ[“RANK”]进程的序号，一般是1个gpu对应一个进程
    #
    # 作者：梯度不下降
    # 链接：https://zhuanlan.zhihu.com/p/885661186
    # 来源：知乎
    # 著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
    if "WORLD_SIZE" in os.environ:
        world_size = int(os.getenv("WORLD_SIZE"))
    else:
        world_size = 1


    if "LOCAL_RANK" in os.environ:
        rank = int(os.getenv("LOCAL_RANK"))
    elif "RANK" in os.environ:
        rank = os.getenv("RANK")v
    else:
        rank = 0

    if rank == 0:
        print(f'DEVICE_COUNT: {torch.cuda.device_count()}')

    torch.manual_seed(123)
    num_epochs = 3
    main(rank, world_size, num_epochs)