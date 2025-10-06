import argparse

from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from  torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,32,3,1)
        self.conv2 = nn.Conv2d(32,64,3,1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1) # 记得从第一维开始展开
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for bs_id, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        o = model(data)
        loss = F.nll_loss(o, target)
        loss.backward()
        optimizer.step()
        if bs_id % args.log_interval == 0:
            print(f'Train Epoch:{epoch}/{args.epoches}  {bs_id * len(data)}/{len(train_loader.dataset)} {100.*bs_id*len(train_loader.dataset)}, loss:{loss.item()}')
            if args.dry_run:
                break
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            o = model(data)
            test_loss += F.nll_loss(o, target, reduction='sum').item()
            pred = o.argmax(dim=1, keepdims=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(f'Test set: average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100*correct/len(test_loader.dataset):.0f})')



def main():
    parser = argparse.ArgumentParser(description="mnist Exam")
    parser.add_argument('--bs', type=int, default=64, metavar='N')
    parser.add_argument('--test-bs', type=int, default=100, metavar='N')
    parser.add_argument('--epoches', type=int, default=14, metavar='N')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='N')
    parser.add_argument('--no-accel', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-model', action='store_true',)

    args = parser.parse_args()
    print(f'{args=}')
    use_accel = not args.no_accel # no_accel=False
    # use_accel = not args.no_accel and torch.accelerator.is_available()
    print(f'使用accel: {use_accel}')
    torch.manual_seed(args.seed)

    if use_accel:
        # device = torch.accelerator.current_accelerator()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'{device=}')
    train_kw = {'batch_size': args.bs}
    test_kw = {'batch_size': args.test_bs}
    if use_accel:
        accel_kw = {
            'num_workers': 1,
            'persistent_workers': True,
            'pin_memory': True,
            'shuffle': True
        }
        train_kw.update(accel_kw)
        test_kw.update(accel_kw)

    transforme = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.137,), (0.3081,))
    ])
    ds1 = datasets.MNIST('../data', train=True, download=True, transform=transforme)
    ds2 = datasets.MNIST('../data', train=False, transform=transforme)

    train_loader = torch.utils.data.DataLoader(ds1, **train_kw)
    test_loader = torch.utils.data.DataLoader(ds2, **test_kw)

    model=Net().to(device)

    optimizer=optim.Adadelta(model.parameters(), lr=args.lr)
    schdeuler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epoches+1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        schdeuler.step()

    if args.save_model:
        torch.save(model.state_dict(), 'minist_cnn.pt')

if __name__ == '__main__':
    main()
