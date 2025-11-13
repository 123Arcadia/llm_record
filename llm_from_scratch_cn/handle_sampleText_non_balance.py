
"""
对欠采样处理
"""
import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import Dataset

data_file_path = '*********'
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
# 原本''spam'数量== 747, 'ham'=4825
def create_balanced_dataset(df: DataFrame):
    # spam实例数量
    num_spam = df[df['Label'] == 'spam'].shape(0)
    # 随机采样'ham'实例以匹配'spam‘实例数量, '123'是随机种子
    ham_subset = df[df['Label'] == 'ham'].sample(num_spam, random_state=123)
    # 'ham'子集和'spam'实例拼接
    balanced_df = pd.concat([ham_subset, df[df['Label'] == 'spam']])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df["Label"].value_counts())
# Label
# ham     747
# spam    747
# Name: count, dtype: int64

# 接下来，我们将字符串类标签"ham"和"spam"更改为整数类标签0和1：
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# 数据集随机划分为训练、验证和测试子集
def random_split(df: DataFrame, train_frac, validation_frac):
    """
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    :param df:
    :param train_frac:
    :param validation_frac:
    :return:
    """
    # 打乱整个 DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    # 计算切分索引.
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 切分 DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


# 下面的SpamDataset类标识训练数据集中最长的序列，并将填充标记添加到其他序列中以匹配该序列长度
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 预标记文本
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果序列长于 max_length，则截断序列
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 将序列填充到最长序列
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length