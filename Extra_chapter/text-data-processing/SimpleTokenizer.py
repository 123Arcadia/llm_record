import re
from collections import defaultdict
from typing import List, Dict

import tiktoken


class SimpleTokenizer:
    def __init__(self, text=None):
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}  # {次数: token}
        self.vocab_size = 0
        self.sepcial_tokens = {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3
        }
        if text:
            self.build_vocab(text)

    def tokenize(self, text: str) -> List:
        # 保留空白字符的分词
        tokens = re.findall(r'\S+|\s+', text)
        return tokens

    def build_vocab(self, text):
        """
        构建词汇表
        :param text:
        :return:
        """
        self.vocab = self.sepcial_tokens.copy()
        self.vocab_size = len(self.sepcial_tokens)
        tokens = self.tokenize(text)
        token_counts = defaultdict(int)
        for token in tokens:
            token_counts[token] += 1
        # 先按照次数降序，在按照key升序
        sorted_tokens = sorted(token_counts.items(), key=lambda x: (-x[1], x[0]))
        for token, _ in sorted_tokens:
            if token not in self.vocab:
                self.vocab[token] = self.vocab_size
                self.vocab_size += 1
        # 如果次数相同，会覆盖
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text):
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

    def decode(self, token_ids: List[int]):
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                tokens.append(self.reverse_vocab[token_id])
            else:
                tokens.append('<unk>')
        text = ''
        prev_token = None
        for token in tokens:
            if prev_token and prev_token.isspace():
                text += token
            else:
                # 如果上一个不是空格
                text = text.rstrip() + token
            prev_token = token
        return text


class BPETokenizer:
    """基于字节对编码(BPE)的分词器"""

    def __init__(self, model_name="gpt2"):
        """
        初始化BPE分词器
        model_name: 模型名称，如"gpt2"或"cl100k_base"(OpenAI的text-embedding-ada-002使用)
        """
        self.encoder = tiktoken.get_encoding(model_name)
        self.vocab_size = self.encoder.n_vocab

    def tokenize(self, text):
        """将文本转换为BPE标记列表"""
        token_ids = self.encoder.encode(text)
        tokens = [self.encoder.decode_single_token_bytes(token_id) for token_id in token_ids]
        return tokens

    def encode(self, text):
        """将文本转换为BPE标记ID序列"""
        return self.encoder.encode(text)

    def decode(self, token_ids):
        """将BPE标记ID序列转换回文本"""
        return self.encoder.decode(token_ids)


if __name__ == "__main__":
    sample_text = "Hello, how are you today? I hope you are doing well."
    print("=== 简单分词器 ===")
    simple_tokenizer = SimpleTokenizer(sample_text)
    tokens = simple_tokenizer.tokenize(sample_text)
    print(f"分词结果: {tokens}")
    encoded = simple_tokenizer.encode(sample_text)
    print(f"编码结果: {encoded}")
    decoded = simple_tokenizer.decode(encoded)
    print(f"解码结果: {decoded}")
    print(f"词汇表大小: {simple_tokenizer.vocab_size}")
    print("\n=== BPE分词器 (GPT-2) ===")
    bpe_tokenizer = BPETokenizer("gpt2")
    bpe_tokens = bpe_tokenizer.tokenize(sample_text)
    print(f"BPE分词结果: {bpe_tokens}")
    bpe_encoded = bpe_tokenizer.encode(sample_text)
    print(f"BPE编码结果: {bpe_encoded}")
    bpe_decoded = bpe_tokenizer.decode(bpe_encoded)
    print(f"BPE解码结果: {bpe_decoded}")
    print(f"BPE词汇表大小: {bpe_tokenizer.vocab_size}")

    # === 简单分词器 ===
    # 分词结果: ['Hello,', ' ', 'how', ' ', 'are', ' ', 'you', ' ', 'today?', ' ', 'I', ' ', 'hope', ' ', 'you', ' ', 'are', ' ', 'doing', ' ', 'well.']
    # 编码结果: [7, 4, 11, 4, 5, 4, 6, 4, 12, 4, 8, 4, 10, 4, 6, 4, 5, 4, 9, 4, 13]
    # 解码结果: Hello, how are you today? I hope you are doing well.
    # 词汇表大小: 14
    #
    # === BPE分词器 (GPT-2) ===
    # BPE分词结果: [b'Hello', b',', b' how', b' are', b' you', b' today', b'?', b' I', b' hope', b' you', b' are', b' doing', b' well', b'.']
    # BPE编码结果: [15496, 11, 703, 389, 345, 1909, 30, 314, 2911, 345, 389, 1804, 880, 13]
    # BPE解码结果: Hello, how are you today? I hope you are doing well.
    # BPE词汇表大小: 50257