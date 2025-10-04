from logging import critical
from typing import Dict, List, Optional, Tuple

import torch

from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class Vocabulary:
    def __init__(self, special_tokens: Optional[Dict[str, int]] = None):
        """
        初始化
        :param special_tokens:
        """
        self.token_to_idx = special_tokens or {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3
        }
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)

    def build_from_texts(self, texts: List[List[str]]) -> None:
        for text in texts:
            for token in text:
                self.add_token(token)

    def add_token(self, token: str)->int:
        """
        添加token
        :param token:
        :return: 该token的标记
        """
        if token not in self.token_to_idx:
            self.token_to_idx[token] = self.vocab_size
            self.idx_to_token[self.vocab_size] = token
            self.vocab_size += 1
        return self.token_to_idx[token]

    def encode(self, text: List[str]) -> List[int]:
        return [self.token_to_idx.get(w, self.token_to_idx["<unk>"]) for w in text]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.idx_to_token.get(idx, "<unk>") for idx in ids]

class CBOWDataset(Dataset):
    def __init__(self, texts: List[List[str]], vocab: Vocabulary, context_size: int = 2):
        self.context_size = context_size
        self.vocab = vocab
        self.data = [] # (上下文(不包含中心词)，中心词)

        for text in texts:
            encoded_text = vocab.encode(text)
            # 在encoded_text中收集左右上下文(前后一定步数的word)
            for i in range(context_size, len(encoded_text) - context_size):
                context = []
                for j in range(-context_size, context_size + 1):
                    if j != 0:
                        context.append(encoded_text[i+j])
                target = encoded_text[i] # 中心词
                self.data.append((context, target))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        context, target = self.data[item]
        return torch.tensor(context,  dtype=torch.long), torch.tensor(target,  dtype=torch.long)


class CBOW(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """

        :param vocab_size:
        :param embedding_dim: 嵌入维度
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim) # 会根据输入调整，但第二维度embedding_dim不会变
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        :param inputs: 输入[batch_size, context_size*2]
        :return: 输出张量 [batch_size, vocab_size]
        """
        # 上下文的嵌入
        embeds = self.embeddings(inputs)
        context_mean = torch.mean(embeds, dim = 1)
        output = self.linear(context_mean)
        return output



def train_cbow_model(texts: List[List[str]],
                     embedding_dim: int = 100,
                     context_size: int = 2,
                     epoches: int = 10,
                     batch_size: int = 32,
                     lr: float = 0.01) -> nn.Embedding:
    """
    续联CBOW模型返回词嵌入
    :param texts:
    :param embedding_dim:
    :param context_size: 上下文大小
    :param epoches:
    :param batch_size:
    :param lr:
    :return:
    """
    # 构建词汇表
    vocab = Vocabulary()
    vocab.build_from_texts(texts)

    # 创建数据集和加载器
    dataset = CBOWDataset(texts, vocab, context_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'{dataset.data=} {len(dataset)=}')

    # 初始化模型、损失函数和优化器
    model = CBOW(vocab.vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练模型
    for epoch in range(epoches):
        total_loss = 0
        for context, target in dataloader:
            output = model(context)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch+1}/{epoches}, loss: {total_loss/len(dataloader):.4f}')

    return model.embeddings

def test_word_embedding():
    print(f'\n============ 测试词嵌入 ============')

    # 示例文本
    texts = [
        ["I", "like", "to", "play", "football"],
        ["Football", "is", "a", "popular", "sport"],
        ["I", "enjoy", "watching", "football", "matches"],
        ["Do", "you", "play", "any", "sports"],
        ["Sports", "are", "good", "for", "health"]
    ]

    # 训练CBOW模型获取词嵌入
    embedding_dim = 10
    context_size = 2
    embeddings = train_cbow_model(
        texts = texts,
        embedding_dim=embedding_dim,
        context_size=context_size,
        epoches=50,
        batch_size=4,
        lr = 0.01
    )

    # 获取词汇表
    vocab = Vocabulary()
    vocab.build_from_texts(texts)

    # 测试词嵌入查找s
    test_words = ["I", "football", "sports", "unknown"]
    print(f'\n词嵌入示例')
    for word in test_words:
        word_id = vocab.encode([word])[0]
        word_vector = embeddings(torch.tensor(word_id, dtype=torch.long)).detach().numpy()
        print(f'{word}: {word_vector[:5]}... (shape:  {word_vector.shape})')

    # 计算词之间相似度分析:
    print(f'\n相似度分析:')
    targets_words = ["football", "sports", "play"]
    for target in targets_words:
        target_id = vocab.encode([target])[0]
        target_vector = embeddings(torch.tensor(target_id, dtype=torch.long))
        # print(f'{target_id}: {target_vector[:5]}... (shape:  {target_vector.shape})')

        print(f'\n与{target}最相似的词:')
        similarities = []
        for word, idx in vocab.token_to_idx.items():
            if word in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                continue
            word_vector = embeddings(torch.tensor(idx, dtype=torch.long))
            # 计算余弦相似度
            sim = torch.cosine_similarity(target_vector.unsqueeze(0),
                                         word_vector.unsqueeze(0)).item()
            similarities.append((word, sim))
        # 按照相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 前3个相似词
        for word, sim in similarities[:3]:
            print(f'    {word}: {sim:.4f}')




if __name__ == '__main__':
    test_word_embedding()
    # ============ 测试词嵌入 ============
    # dataset.data=[([4, 5, 7, 8], 6), ([9, 10, 12, 13], 11), ([4, 14, 8, 16], 15), ([17, 18, 19, 20], 7), ([21, 22, 24, 25], 23)] len(dataset)=5
    # Epoch: 1/50, loss: 3.1324
    # Epoch: 2/50, loss: 3.1630
    # Epoch: 3/50, loss: 3.0487
    # Epoch: 4/50, loss: 2.9380
    # Epoch: 5/50, loss: 2.7755
    # Epoch: 6/50, loss: 2.7152
    # Epoch: 7/50, loss: 2.5232
    # Epoch: 8/50, loss: 2.5886
    # Epoch: 9/50, loss: 2.4937
    # Epoch: 10/50, loss: 2.2531
    # Epoch: 11/50, loss: 1.9360
    # Epoch: 12/50, loss: 2.0573
    # Epoch: 13/50, loss: 2.1152
    # Epoch: 14/50, loss: 1.9265
    # Epoch: 15/50, loss: 1.9247
    # Epoch: 16/50, loss: 1.6685
    # Epoch: 17/50, loss: 1.5961
    # Epoch: 18/50, loss: 1.5594
    # Epoch: 19/50, loss: 1.5487
    # Epoch: 20/50, loss: 1.3765
    # Epoch: 21/50, loss: 1.3630
    # Epoch: 22/50, loss: 1.0145
    # Epoch: 23/50, loss: 1.1796
    # Epoch: 24/50, loss: 0.8791
    # Epoch: 25/50, loss: 1.0050
    # Epoch: 26/50, loss: 0.8369
    # Epoch: 27/50, loss: 0.8571
    # Epoch: 28/50, loss: 0.6991
    # Epoch: 29/50, loss: 0.6025
    # Epoch: 30/50, loss: 0.5562
    # Epoch: 31/50, loss: 0.5215
    # Epoch: 32/50, loss: 0.4719
    # Epoch: 33/50, loss: 0.4290
    # Epoch: 34/50, loss: 0.4008
    # Epoch: 35/50, loss: 0.4362
    # Epoch: 36/50, loss: 0.3422
    # Epoch: 37/50, loss: 0.2994
    # Epoch: 38/50, loss: 0.2934
    # Epoch: 39/50, loss: 0.3114
    # Epoch: 40/50, loss: 0.2364
    # Epoch: 41/50, loss: 0.2381
    # Epoch: 42/50, loss: 0.3515
    # Epoch: 43/50, loss: 0.2582
    # Epoch: 44/50, loss: 0.1762
    # Epoch: 45/50, loss: 0.1915
    # Epoch: 46/50, loss: 0.2586
    # Epoch: 47/50, loss: 0.2355
    # Epoch: 48/50, loss: 0.1508
    # Epoch: 49/50, loss: 0.1226
    # Epoch: 50/50, loss: 0.1754
    #
    # 词嵌入示例
    # I: [-1.0948917  -0.60530263 -0.12006583  0.3609816   1.1901138 ]... (shape:  (10,))
    # football: [ 1.6599002  -1.6759335  -1.4154464  -0.16711393  0.5173793 ]... (shape:  (10,))
    # sports: [-0.12306848  0.83442396 -3.099877    1.5852575  -0.20742619]... (shape:  (10,))
    # unknown: [ 0.55423677  0.20594753  0.3934137  -0.4609817   0.44276828]... (shape:  (10,))
    #
    # 相似度分析:
    #
    # 与football最相似的词:
    #     football: 1.0000
    #     Football: 0.7100
    #     enjoy: 0.6668
    #
    # 与sports最相似的词:
    #     sports: 1.0000
    #     health: 0.5230
    #     to: 0.5059
    #
    # 与play最相似的词:
    #     play: 1.0000
    #     to: 0.6955
    #     like: 0.6949