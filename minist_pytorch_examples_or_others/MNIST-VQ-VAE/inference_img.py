"""
推理重建图像
"""

import torch
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from vq_vae import MNIST_VQVAE


device = "cuda" if torch.cuda.is_available() else "cpu"




def main():
    train_dataset = datasets.MNIST(root="../data", train=False, transform=transforms.Compose([transforms.ToTensor(),]))
    img, label = train_dataset[0]
    x = img.unsqueeze(0).to(device)
    vqvae_model = MNIST_VQVAE(codebook_size=64).to(device)

    recon_x, ze,zq = vqvae_model(x)
    print(f'{img.detach()=} ')
    print(f"{img.shape=}") # img.shape=torch.Size([1, 28, 28])
    print(f'{x=}')
    print(f'{x.shape=}') # x.shape=torch.Size([1, 1, 28, 28])
    print(f'{recon_x.shape=}') # recon_x.shape=torch.Size([1, 1, 28, 28])

    plt.figure(figsize=(2,2))
    ax1 = plt.subplot(1,2,1)
    ax1.set_title("recon img")
    plt.imshow(recon_x.detach().cpu().squeeze().numpy())
    ax2 = plt.subplot(1,2,2)
    ax2.set_title("ori img")
    plt.imshow(img.detach().cpu().squeeze().numpy())
    plt.show()

    _, code_idx = vqvae_model.encode(x)
    print(f'图像离散编码:{code_idx.squeeze()}')
    # 图像离散编码:tensor([[16, 16, 16, 16, 16, 16, 16],
    #         [16, 16, 20, 16, 16, 16, 16],
    #         [16, 16, 16, 16, 16, 20, 16],
    #         [16, 16, 16, 16, 16, 16, 16],
    #         [16, 16, 16, 16, 16, 16, 16],
    #         [16, 16, 16, 16, 16, 16, 16],
    #         [16, 16, 16, 16, 16, 16, 16]], device='cuda:0')


if __name__ == '__main__':
    main()