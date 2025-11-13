import os

import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, w):
        if w not in self.word2idx:
            self.idx2word.append(w)
            self.word2idx[w] = len(self.idx2word) - 1
        return self.word2idx[w]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        # print(f'{path=}')
        self.dictionary = Dictionary()
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))

    def tokenize(self, path):
        # 统计
        print(f'{path=}')
        assert os.path.exists(path), "This path is not exist!"
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                # if line is None or line == "\n":
                #     continue
                # new_l = line.split()
                # if new_l[0] == '=' and new_l[-1] == '=':
                #     continue
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

        # for w, id in self.dictionary.word2idx.items():
        #     print(f'{w} ----- {id=}')
        # 拼接所有的ids成一个tensor
        with open(path, 'r', encoding='utf-8') as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                # print(f'{words=}')

                ids = []
                for w in words:
                    ids.append(self.dictionary.word2idx[w])

                idss.append(torch.tensor(ids).type(torch.int64))
                # print(f'{words=}')
                # print(f'{self.dictionary.word2idx["<eos>"]=}')

            # print('-'*50)
            # print(f'{ids=}')
            # print(f'{len(idss)=}')
            # # print(f'{[i.shape for i in idss]}')
            # print(f'{words=}')
            # print(f'{self.dictionary.word2idx[words[0]]=}')
            ids = torch.cat(idss)

        return ids
