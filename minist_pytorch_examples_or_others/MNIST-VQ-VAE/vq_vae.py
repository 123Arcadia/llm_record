import torch
from importlib.metadata import version
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

pkgs = ["matplotlib", "numpy", "torch"]


class MNIST_VQVAE(torch.nn.Module):
    def __init__(self, codebook_size):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU()
        )  # (B,C=32,H=7,W=7)
        self.codebook = torch.nn.Parameter(torch.randn(codebook_size, 32))  # (CODEBOOK_SIZE,32)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )  # (B,C=1,H=28,W=28)

    def encode(self, x):
        # 图像压缩
        ze = self.encoder(x)  # ze=(B,C=32,H=7,W=7)
        # VQ-VAE量化
        ze_extended = ze.unsqueeze(1)  # (B,1,C=32,H=7,W=7)
        codebook_extended = self.codebook.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1,CODEBOOK_SIZE,C=32,H=1,W=1)
        dist = (ze_extended - codebook_extended) ** 2
        dist = dist.sum(dim=2)
        code_idx = dist.argmin(1)  # 取最邻近codebook下标, shape=(B,H=7,W=7)
        return ze, code_idx

    def forward(self, x):  # x: (B,C=1,H=28,W=28)
        # 图像压缩&离散编码
        ze, code_idx = self.encode(x)
        # 离线编码转稠密码本向量
        zq = self.codebook[code_idx]  # 取codebook的embedding, shape=(B,H=7,W=7,C=32)
        zq = zq.permute(0, 3, 1, 2)  # zq=(B,C=32,H=7,W=7)
        # 图像解压
        x_recon = self.decoder(ze + (zq - ze).detach())  # x_recon=(B,C=1,H=28,W=28)
        return x_recon, ze, zq

def main():
    print(f'设备: {device}')
    for p in pkgs:
        print(f"{p} version: {version(p)}")

    # train VA_VAE
    vqvae_model = MNIST_VQVAE(codebook_size=64).to(device)
    optimizer = optim.AdamW(vqvae_model.parameters(), lr=1e-3)
    train_dataset = datasets.MNIST(root="../data", train=True, transform=transforms.Compose([transforms.ToTensor(), ])
                                   ,download=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

    alpha = 0.25  # encoder的部分
    beta = 1
    epoches = 50
    for epoch in range(epoches):
        for batch_idx, (batch_img, batch_label) in enumerate(train_loader):
            batch_img = batch_img.to(device)
            recon_img, ze, zq = vqvae_model(batch_img)
            loss = ((recon_img-batch_img).pow(2).mean() + alpha * (ze - zq.detach()).pow(2).mean()
                                                        + beta * (ze.detach() - zq).pow(2).mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch: {epoch}/{epoches}, Loss: {loss.item():.4f}")

if __name__ == '__main__':
    main()
