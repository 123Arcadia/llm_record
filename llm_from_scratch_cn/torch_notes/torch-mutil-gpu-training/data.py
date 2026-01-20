from torchvision import datasets, transforms

file_path = '../../../minist_pytorch_examples_or_others/data'
train_dataset = datasets.MNIST(root=file_path, train=True, transform=transforms.ToTensor(), download=False)
test_dataset = datasets.MNIST(root=file_path, train=False, transform=transforms.ToTensor(), download=False)
