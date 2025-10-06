
import argparse

import torch
import torch.utils.data
import tqdm
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

"""
变分自编码器(Variational auto-encoder,VAE

"""

parser = argparse.ArgumentParser(description="vae MNIST Example")

parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--epoches", type=int, default=10)
parser.add_argument("--no-accel", action='store_true',
                    help="disables accelerator")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--log-interval', type=int, default=10)

args = parser.parse_args()

# use_accel = not args.no_accel and torch.accelerator.is_available()
# torch.manual_seed(args.seed)
#
# if use_accel:
#     device = torch.accelerator.current_accelerator()
# else:
#     device = torch.device("cpu")

use_accel = False
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f'{device=}')

kw = {
    'num_workers': 1,
    'pin_memory': True,
} if use_accel else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../minist_pytorch_examples/data", train=True, download=True, transform=transforms.ToTensor()),
    batch_size=args.bs, shuffle=True, **kw
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../minist_pytorch_examples/data", train=False, download=False, transform=transforms.ToTensor()),
    batch_size=args.bs, shuffle=False, **kw
)


class VAE(nn.Module):
    """
    VAE详解: https://blog.csdn.net/weixin_42491648/article/details/132384913
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        """
        :param x:
        :return: (均值，方差)
        """
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def loss_func(recon_x, x, mu, logvar):
    # 就是mse(因为上面经过了sigmod，这里直接用binary_cross_entropy而不是binary_cross_entropy_with_logits)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD + BCE


def train(epoch):
    model.train()
    train_loss = 0
    for b_id, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_func(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if b_id % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, b_id * len(data), len(train_loader.dataset),
                       100. * b_id / len(train_loader),
                       loss.item() / len(data)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            print(f'{i=} {data.shape=}\t{recon_batch.shape=}\t{mu.shape=}\t{logvar.shape=}')
            # data.shape=torch.Size([128, 1, 28, 28])
            # recon_batch.shape=torch.Size([128, 784])
            # mu.shape=torch.Size([128, 20])
            # logvar.shape=torch.Size([128, 20])
            # i=78 data.shape=torch.Size([16, 1, 28, 28])	recon_batch.shape=torch.Size([16, 784])	mu.shape=torch.Size([16, 20])	logvar.shape=torch.Size([16, 20])
            test_loss += loss_func(recon_batch, data, mu, logvar)
            if i in [1, 20, 50, len(test_loader.dataset)]:
                n = min(data.size(0), 8)
                comparison = torch.cat([data,
                                        recon_batch.view(args.bs, 1, 28, 28)])
                # comparison = torch.cat([data[:n],
                #                         recon_batch.view(args.bs, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           './result/reconstruction_' + str(i) + '.png', nrow=n)
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == '__main__':
    for epoch in tqdm.tqdm(range(1, args.epoches + 1)):
        train(epoch)
        print('-'*50)
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                       '../output/sample_' + str(epoch) + '.png')
