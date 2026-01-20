import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"  # 必须在`import torch`语句之前设置才能生效

import torch
from torch.utils.data import DataLoader
from data import train_dataset
from model import Net
from plot_losses import plot_losses


def main():


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bsz = 64
    train_loader = DataLoader(train_dataset, batch_size=bsz, shuffle=True)

    model = Net()
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    losses = []

    f_name = f"{__file__}".split(".")[0]
    # f_name = f"{__name__}"

    for i, (input, labels) in enumerate(train_loader):
        input = input.to(device)
        labels = labels.to(device)
        outs = model(input, labels=labels)
        loss = outs[0]  # loss
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f'i: {i}, loss: {loss.item():.4f}')



    plot_losses(losses, f"{f_name}_loss.jpg")
    print(os.getenv("CUDA_VISIBLE_DEVICES"))
    # 2,3

if __name__ == '__main__':
    main()

