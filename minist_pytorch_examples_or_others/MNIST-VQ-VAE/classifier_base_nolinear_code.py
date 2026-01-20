import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets
from torchvision.transforms import transforms
from vq_vae import MNIST_VQVAE

device = "cuda" if torch.cuda.is_available() else "cpu"
vqvae_model = MNIST_VQVAE(codebook_size=64).to(device)


class MNISTClassifier(nn.Module):
    def __init__(self, codebook_size):
        super().__init__()
        self.emb = nn.Embedding(codebook_size, 32)
        self.classify = nn.Sequential(
            nn.Linear(7 * 7 * 32, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        z = self.emb(x)  # 32  shape=(B,H,W,32)
        z = z.view(z.size(0), -1)  # shape=(B, H*W*32)
        return self.classify(z)  # (B, 10)


def tokenizer(batch):
    batch_img = [img for img, _ in batch]
    batch_label = [label for _, label in batch]
    batch_label = torch.tensor(batch_label, dtype=torch.long)
    batch_img = torch.stack(batch_img).to(device)
    vqvae_model.eval()
    with torch.no_grad():
        _, token_id = vqvae_model.encode(batch_img)
    return token_id.detach(), batch_label

def test():
    """
    验证分类模型
    """
    test_dataset = datasets.MNIST(root='../data', train=False, download=False,
                                  transform=transforms.Compose([transforms.ToTensor(), ]))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, collate_fn=tokenizer)

    correct = 0
    total = 0

    classifier_model = MNISTClassifier(codebook_size=64).to(device=device)
    for batch_idx, (batch_img_token, batch_label) in enumerate(test_dataloader):
        batch_img_token = batch_img_token.to(device)
        batch_label = batch_label.to(device)

        logits = classifier_model(batch_img_token)
        pred_label = torch.argmax(logits, dim=1) # 通道层级
        correct += (pred_label == batch_label).sum()
        total += len(batch_label)
    print(f'{total=}, {correct.item()=}, acc: {correct/total:.4f}')
    # total=10000, correct=tensor(974, device='cuda:0'), acc: 0.0974


def main():
    train_dataset = datasets.MNIST(root='../data', train=True, download=False,
                                   transform=transforms.Compose([transforms.ToTensor(), ]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=tokenizer)

    classifier_model = MNISTClassifier(codebook_size=64).to(device=device)
    optimizer = optim.AdamW(classifier_model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    epoches = 50
    for epoch in range(epoches):
        for batch_idx, (batch_img_token, batch_label) in enumerate(train_dataloader):
            batch_img_token = batch_img_token.to(device)
            batch_label = batch_label.to(device)

            logits = classifier_model(batch_img_token)
            loss = loss_fn(logits, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch:}/{epoches}, Loss: {loss.item():.4f}')

if __name__ == '__main__':
    # main()
    test()
