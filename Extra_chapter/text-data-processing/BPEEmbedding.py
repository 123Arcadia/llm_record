import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import tiktoken
from pycparser.ply.yacc import token
from sympy.physics.secondquant import wicks


class SampleBPETokenizer:
    """
    简单实现BPE
    """

    def __init__(self, vocab_size: int = 100):
        """
        初始化
        :param vocab_size: 词汇表大小
        """
        self.vocab_size = vocab_size
        self.vocab = {}  # 词汇表: 标记 -> ID
        self.reverse_vocab = {}
        self.bpe_ranks = {}  # bpe合并规则优先级
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<bos>": 2,
            "<eos>": 3,
        }
        self.vocab_size_actual = len(self.special_tokens)

        for token, idx in self.special_tokens.items():
            self.vocab[token] = idx
            self.reverse_vocab[idx] = token

    def build_vocab(self, texts: List[str]):
        """
        构建词汇表
        :param texts:
        :return:
        """
        # 初始化词汇表，包含所有的单字符
        token_counts = Counter()
        # token_counts_2 = Counter()
        for text in texts:
            words = [' '.join(list(text))]
            # print(f'{words=}')
            for w in words:
                # token_counts_2.update(w)
                token_counts[w] += 1

        """
        这两个结果的区别：
        words=['H e l l o ,   h o w   a r e   y o u   t o d a y ?']
        words=['I   h o p e   y o u   a r e   d o i n g   w e l l .']
        words=['T h i s   i s   a   t e s t   o f   t h e   B P E   t o k e n i z e r .']
        words都是一个list，包含一个字符串
        Counter().update()方法是接收一个【可迭代对象】来统计字符出现次数
        Counter()[w]是把字符串当成一个整对象来统计的，它不会进一步划分字符
        
        """
        # print(f'{token_counts=}')
        # token_counts=Counter({'H e l l o ,   h o w   a r e   y o u   t o d a y ?': 1, 'I   h o p e   y o u   a r e   d o i n g   w e l l .': 1, 'T h i s   i s   a   t e s t   o f   t h e   B P E   t o k e n i z e r .': 1})
        # print(f'{token_counts_2=}')
        # token_counts_2=Counter({' ': 100, 'e': 9, 'o': 9, 't': 5, 'l': 4, 'h': 4, 'a': 4, 'i': 4, 'r': 3, 'y': 3, 's': 3, 'w': 2, 'u': 2, 'd': 2, 'n': 2, '.': 2, 'H': 1, ',': 1, '?': 1, 'I': 1, 'p': 1, 'g': 1, 'T': 1, 'f': 1, 'B': 1, 'P': 1, 'E': 1, 'k': 1, 'z': 1})
        # 统计初始的字符词汇表
        chars = set()
        for text in texts:
            chars.update(text)

        # 添加字符到词汇表
        for char in sorted(chars):
            if char not in self.vocab:
                self.vocab[char] = self.vocab_size_actual
                self.reverse_vocab[self.vocab_size_actual] = char
                self.vocab_size_actual += 1

        # BPE合并
        num_merges = self.vocab_size - self.vocab_size_actual
        # print(f'{token_counts=}\n{num_merges=}')
        # token_counts=Counter({' ': 100, 'e': 9, 'o': 9, 't': 5, 'l': 4, 'h': 4, 'a': 4, 'i': 4, 'r': 3, 'y': 3, 's': 3, 'w': 2, 'u': 2, 'd': 2, 'n': 2, '.': 2, 'H': 1, ',': 1, '?': 1, 'I': 1, 'p': 1, 'g': 1, 'T': 1, 'f': 1, 'B': 1, 'P': 1, 'E': 1, 'k': 1, 'z': 1})
        # num_merges=17
        # print(f'{self.vocab=}')
        # self.vocab={'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, ' ': 4, ',': 5, '.': 6, '?': 7, 'B': 8, 'E': 9, 'H': 10, 'I': 11, 'P': 12, 'T': 13, 'a': 14, 'd': 15, 'e': 16, 'f': 17, 'g': 18, 'h': 19, 'i': 20, 'k': 21, 'l': 22, 'n': 23, 'o': 24, 'p': 25, 'r': 26, 's': 27, 't': 28, 'u': 29, 'w': 30, 'y': 31, 'z': 32}
        # print(f'{self.vocab_size=}')
        # self.vocab_size=50
        # print(f'{self.vocab_size_actual=}')
        # self.vocab_size_actual=33
        # assert 0, "暂停!!!!"
        if num_merges <= 0:
            return

        pairs = token_counts.copy()
        # print(f'{pairs=}')
        # pairs=Counter({'H e l l o ,   h o w   a r e   y o u   t o d a y ?': 1, 'I   h o p e   y o u   a r e   d o i n g   w e l l .': 1, 'T h i s   i s   a   t e s t   o f   t h e   B P E   t o k e n i z e r .': 1})
        for i in range(num_merges):
            # 计算所有相邻字节对的频率
            stats = self._get_stats(pairs)
            # i=0 stats=defaultdict(<class 'int'>, {('H', 'e'): 1, ('e', 'l'): 2, ('l', 'l'): 2, ('l', 'o'): 1, ('o', ','): 1, (',', 'h'): 1, ('h', 'o'): 2, ('o', 'w'): 1, ('w', 'a'): 1, ('a', 'r'): 2, ('r', 'e'): 2, ('e', 'y'): 2, ('y', 'o'): 2, ('o', 'u'): 2, ('u', 't'): 1, ('t', 'o'): 3, ('o', 'd'): 1, ('d', 'a'): 1, ('a', 'y'): 1, ('y', '?'): 1, ('I', 'h'): 1, ('o', 'p'): 1, ('p', 'e'): 1, ('u', 'a'): 1, ('e', 'd'): 1, ('d', 'o'): 1, ('o', 'i'): 1, ('i', 'n'): 1, ('n', 'g'): 1, ('g', 'w'): 1, ('w', 'e'): 1, ('l', '.'): 1, ('T', 'h'): 1, ('h', 'i'): 1, ('i', 's'): 2, ('s', 'i'): 1, ('s', 'a'): 1, ('a', 't'): 1, ('t', 'e'): 1, ('e', 's'): 1, ('s', 't'): 1, ('o', 'f'): 1, ('f', 't'): 1, ('t', 'h'): 1, ('h', 'e'): 1, ('e', 'B'): 1, ('B', 'P'): 1, ('P', 'E'): 1, ('E', 't'): 1, ('o', 'k'): 1, ('k', 'e'): 1, ('e', 'n'): 1, ('n', 'i'): 1, ('i', 'z'): 1, ('z', 'e'): 1, ('e', 'r'): 1, ('r', '.'): 1})
            if not stats:
                break

            # 选择最频繁的字节对
            best = max(stats, key=stats.get)
            # 记录合并规则的优先级
            self.bpe_ranks[best] = i
            # 合并词汇表中的字节对
            pairs = self._merge_vocab(best, pairs)
            # print(f'{best=}')
            # print(f'{pairs=}')

            # 将新合并的标记添加到词汇表
            new_token = ''.join(best)
            # print(f'{new_token=}')
            # print(f'{self.vocab=}')
            # print(f'{self.vocab_size_actual=}')
            if new_token not in self.vocab:
                self.vocab[new_token] = self.vocab_size_actual
                self.reverse_vocab[self.vocab_size_actual] = new_token
                self.vocab_size_actual += 1

    def _get_stats(self, pairs: Dict[Tuple[str, str], int]) -> Dict[Tuple[str, str], int]:
        """计算所有相邻字符的频率"""
        stats = defaultdict(int)
        for word, freq in pairs.items():
            symbols = word.split()  # 如果是单个字符就没用
            for i in range(len(symbols) - 1):
                stats[symbols[i], symbols[i + 1]] += freq
        return stats

    def _merge_vocab(self, pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
        """
        合并最频繁的字节对
        :param pair: 最频繁的字节对
        :param v_in: 已存在的对
        :return:
        """
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """

        :param sample_text:
        :return:
        """
        tokens = self.tokenize(text)
        encoded = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]
        if add_special_tokens:
            encoded = [self.vocab["<bos>"]] + encoded + [self.vocab["<eos>"]]
        return encoded

    def tokenize(self, text: str) -> List[str]:
        """
        分词作为BPE标记
        :param sample_text:
        :return:
        """
        tokens = []
        for token in text.split():
            if token in self.special_tokens:
                tokens.append(token)
            else:
                tokens.extend(self.bpe(token))
        return tokens

    def bpe(self, token: str) -> List[str]:
        """
        对单个分词使用bpe算法
        :param text:
        :return:
        """
        if token in self.special_tokens:
            return [token]
        word = list(token)
        if len(word) == 0:
            return []
        if len(word) == 1:
            return [word[0]]

        pairs = self._get_pairs(word)

        while True:
            # 找到优先级最高的字节对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            # bigram是最频繁的相邻字符对（字节对）
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j]) # # 将 i 到 j 之间的字符加入 new_word
                    i = j
                except:
                    # 未找到 first，将剩余字符加入 new_word 并退出循环
                    new_word.extend(word[i:])
                    break
                # 若 first 后面紧跟 second，则合并为 first+second
                if i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
                #  更新 word 为合并后的结果
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = self._get_pairs(word)
        return word

    def _get_pairs(self, word: List[str]):
        """
        获取单词中所有的相邻标记对
        :param word:
        :return:
        """
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs

    def decode(self, ids: List[int], remove_special_tokens: bool = False) -> List[str]:
        tokens=[]
        for idx in ids:
            if idx in self.reverse_vocab:
                token = self.reverse_vocab[idx]
                if remove_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append("<unk>")
        text = ''.join(tokens)
        return text



def test_tiktoken_bpe():
    """测试tiktoken库的BPE分词器"""
    print("\n=====测试tiktoken BPE分词器=====")
    # 初始化tiktoken bpe分词器
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except KeyError:
        tokenizer = tiktoken.get_encoding("cl100k_base")

    print(f'词汇表大小:{tokenizer.n_vocab}')
    # 测试分词
    sample_text = "Hello, this is a test!"
    tokens = tokenizer.encode(sample_text)
    print(f'\n编码结果(ID) : {tokens}')

    # 转换ID为字节
    token_types = [tokenizer.decode_single_token_bytes(token) for token in tokens]
    print(f'\n解码结果(字节): {token_types}')

    # 解码
    decoded = tokenizer.decode(tokens)
    print(f'\n解码结果: {decoded}')

    # 测试包含未知词汇的文本
    unknown_text = "This is a unicorn 🦄 test."
    encoded_unknown = tokenizer.encode(unknown_text)
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f'含有未知词汇的文本: {unknown_text}')
    print(f'编码: {encoded_unknown}')
    print(f'编解码: {decoded_unknown}')

    # 计算文本的tokens数量
    print(f'\n文本: "{sample_text}" 的数量是 {len(tokens)}')
    # =====测试tiktoken BPE分词器=====
    # 词汇表大小:50257
    #
    # 编码结果(ID) : [15496, 11, 428, 318, 257, 1332, 0]
    #
    # 解码结果(字节): [b'Hello', b',', b' this', b' is', b' a', b' test', b'!']
    #
    # 解码结果: Hello, this is a test!
    # 含有未知词汇的文本: This is a unicorn 🦄 test.
    # 编码: [1212, 318, 257, 44986, 12520, 99, 226, 1332, 13]
    # 编解码: This is a unicorn 🦄 test.
    #
    # 文本: "Hello, this is a test!" 的数量是 7


def test_simple_bpe_tokenizer():
    """测试简单的BPE分词器"""
    print("\n=====测试简单的BPE分词器=====")
    texts = [
        "Hello, how are you today?",
        "I hope you are doing well.",
        "This is a test of the BPE tokenizer."
    ]
    # 初始化BPE分词器
    tokenizer = SampleBPETokenizer(vocab_size=50)

    # 构建词汇表
    tokenizer.build_vocab(texts)
    print(f'词汇表大小: {tokenizer.vocab_size_actual}')
    print(f'前10个词汇项: {list(tokenizer.vocab.items())[:10]}')
    # print(f'{tokenizer.vocab=}')
    # print(f'{tokenizer.vocab_size_actual=}')
    # tokenizer.vocab={'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3, ' ': 4, ',': 5, '.': 6, '?': 7, 'B': 8, 'E': 9, 'H': 10, 'I': 11, 'P': 12, 'T': 13, 'a': 14, 'd': 15, 'e': 16, 'f': 17, 'g': 18, 'h': 19, 'i': 20, 'k': 21, 'l': 22, 'n': 23, 'o': 24, 'p': 25, 'r': 26, 's': 27, 't': 28, 'u': 29, 'w': 30, 'y': 31, 'z': 32, 'to': 33, 'el': 34, 'ell': 35, 'ho': 36, 'ar': 37, 'are': 38, 'yo': 39, 'you': 40, 'is': 41, 'Hell': 42, 'Hello': 43, 'Hello,': 44, 'Hello,ho': 45}
    # tokenizer.vocab_size_actual=46
    # assert  0, "暂停!!!"

    # 测试分词
    sample_text = "Hello, this is a test!"
    tokens = tokenizer.tokenize(sample_text)
    print(f'\n分词结果: {tokens}')

    # 测试编码
    encoded = tokenizer.encode(sample_text, add_special_tokens=True)
    print(f'编码结果: {encoded}')

    # 测试解码
    decoded = tokenizer.decode(encoded, remove_special_tokens=True)
    print(f'解码结果: {decoded}')

    # 测试含有未知词汇的文本
    unknown_text = "This is a unicorn 🦄 test."
    encoded_unknown = tokenizer.encode(unknown_text)
    decoded_unknown = tokenizer.decode(encoded_unknown)
    print(f'\n包含未知词汇的文本: {unknown_text}')
    print(f'编码结果: {encoded_unknown}')
    print(f'解码结果: {decoded_unknown}')

    print(f'测试全部完成!')
    # =====测试简单的BPE分词器=====
    # 词汇表大小: 46
    # 前10个词汇项: [('<pad>', 0), ('<unk>', 1), ('<bos>', 2), ('<eos>', 3), (' ', 4), (',', 5), ('.', 6), ('?', 7), ('B', 8), ('E', 9)]
    #
    # 分词结果: ['Hello,', 't', 'h', 'is', 'is', 'a', 't', 'e', 's', 't', '!']
    # 编码结果: [2, 44, 28, 19, 41, 41, 14, 28, 16, 27, 28, 1, 3]
    # 解码结果: Hello,thisisatest
    #
    # 包含未知词汇的文本: This is a unicorn 🦄 test.
    # 编码结果: [13, 19, 41, 41, 14, 29, 23, 20, 1, 24, 26, 23, 1, 28, 16, 27, 28, 6]
    # 解码结果: Thisisauni<unk>orn<unk>test.
    # 测试全部完成!


if __name__ == '__main__':
    """
    将未见过的单词分解为子词单元
    """
    test_simple_bpe_tokenizer()
    test_tiktoken_bpe()
