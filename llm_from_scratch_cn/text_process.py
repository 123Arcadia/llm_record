import re
from importlib.metadata import version, files

import tiktoken
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

print("torch version:", version("torch"))
# print("torch files:", files("torch"))
print("tiktoken version:", version("tiktoken"))
# torch version: 2.4.1
# tiktoken version: 0.7.0

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        print(f'{len(token_ids)=}') # 5145
        # 使用滑动窗口将图书分块为最大长度的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

        print(f'{len(self.input_ids)=}')
        print(f'{len(self.target_ids)=}') # 1286

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        print(f'{idx=} {type(idx)}')
        # return torch.tensor(self.input_ids)[idx], torch.tensor(self.target_ids)[idx]
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True):
    # 分词器初始化
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    return dataloader


with open('the-verdict.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()

print(f'Total number if character: {len(raw_text)}')




tokenizer = tiktoken.get_encoding("gpt2")
encoded_text = tokenizer.encode(raw_text)

vocab_size = 50257
output_dim = 256
block_size = 1024


token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(block_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length)

for batch in dataloader:
    x, y = batch
    print(f"{batch=}")

    token_embeddings = token_embedding_layer(x)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(f'{token_embeddings.shape=}')
    print(f'{pos_embeddings.shape=}')
    input_embeddings = token_embeddings + pos_embeddings

    break

print(f'{input_embeddings.shape=}')