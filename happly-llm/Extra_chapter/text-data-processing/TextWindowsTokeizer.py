
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional

class TextWindowDataset(Dataset):
    """使用滑动窗口对文本进行采样的数据集"""

    def __init__(self,
                 text: List[int],
                 context_length: int,
                 stride: int = 1,
                 pad_id: int = 0):
        """
        初始化文本窗口数据集

        Args:
            text: 已编码的文本（整数列表）
            context_length: 上下文长度（窗口大小）
            stride: 滑动窗口的步长，默认为1
            pad_id: 填充标记的ID
        """
        self.text = text
        self.context_length = context_length
        self.stride = stride
        self.pad_id = pad_id

        # 计算有效样本数量
        self.num_samples = max(0, (len(text) - context_length) // stride + 1)

    def __len__(self) -> int:
        """返回数据集的样本数量"""
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            元组(inputs, targets)，其中inputs是输入序列，targets是目标序列
        """
        # 计算窗口起始位置
        start = idx * self.stride

        # 确保窗口不超出文本长度
        end = start + self.context_length
        if end > len(self.text):
            # 截取最后可能的有效窗口
            end = len(self.text)
            start = end - self.context_length

        # 提取输入序列和目标序列
        inputs = self.text[start:end]
        targets = self.text[start +1:end +1]  # 目标是输入的下一个标记

        # 如果目标序列长度不足，用pad_id填充
        if len(targets) < self.context_length:
            targets = targets + [self.pad_id] * (self.context_length - len(targets))

        # 转换为张量
        inputs = torch.tensor(inputs, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)

        return inputs, targets


def create_data_loader(text: List[int],
                       context_length: int,
                       batch_size: int,
                       stride: int = 1,
                       shuffle: bool = False) -> DataLoader:
    """
    创建文本窗口数据加载器

    Args:
        text: 已编码的文本（整数列表）
        context_length: 上下文长度
        batch_size: 批次大小
        stride: 滑动窗口步长
        shuffle: 是否打乱数据

    Returns:
        数据加载器
    """
    dataset = TextWindowDataset(
        text=text,
        context_length=context_length,
        stride=stride
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return data_loader


# 测试代码
def test_sliding_window():
    """测试滑动窗口数据采样"""
    print("\n===== 测试滑动窗口数据采样 =====")

    # 示例文本（已编码）
    encoded_text = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    print(f"原始文本: {encoded_text}")

    # 参数设置
    context_length = 4
    stride = 2
    batch_size = 2

    # 创建数据集
    dataset = TextWindowDataset(
        text=encoded_text,
        context_length=context_length,
        stride=stride
    )

    # 打印数据集信息
    print(f"\n数据集大小: {len(dataset)}")

    # 测试获取单个样本
    print("\n--- 测试获取单个样本 ---")
    for i in range(min(3, len(dataset))):
        inputs, targets = dataset[i]
        print(f"样本 {i}:")
        print(f"  输入: {inputs}")
        print(f"  目标: {targets}")

    # 创建数据加载器
    data_loader = create_data_loader(
        text=encoded_text,
        context_length=context_length,
        batch_size=batch_size,
        stride=stride,
        shuffle=False
    )

    # 测试批次数据
    print("\n--- 测试批次数据 ---")
    for i, (batch_inputs, batch_targets) in enumerate(data_loader):
        print(f"批次 {i}:")
        print(f"  输入形状: {batch_inputs.shape}")
        print(f"  输入数据:")
        print(batch_inputs)
        print(f"  目标形状: {batch_targets.shape}")
        print(f"  目标数据:")
        print(batch_targets)

    # 测试不同步长
    print("\n--- 测试不同步长 ---")
    for stride in [1, 2, 3]:
        dataset = TextWindowDataset(
            text=encoded_text,
            context_length=context_length,
            stride=stride
        )
        print(f"步长为 {stride} 时的样本数: {len(dataset)}")

        # 打印前两个样本
        if len(dataset) > 0:
            inputs, targets = dataset[0]
            print(f"  第一个样本输入: {inputs}")
            print(f"  第一个样本目标: {targets}")

        if len(dataset) > 1:
            inputs, targets = dataset[1]
            print(f"  第二个样本输入: {inputs}")
            print(f"  第二个样本目标: {targets}")
if __name__ == "__main__":
    test_sliding_window()
    # ===== 测试滑动窗口数据采样 =====
    # 原始文本: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    #
    # 数据集大小: 4
    #
    # --- 测试获取单个样本 ---
    # 样本 0:
    #   输入: tensor([101, 102, 103, 104])
    #   目标: tensor([102, 103, 104, 105])
    # 样本 1:
    #   输入: tensor([103, 104, 105, 106])
    #   目标: tensor([104, 105, 106, 107])
    # 样本 2:
    #   输入: tensor([105, 106, 107, 108])
    #   目标: tensor([106, 107, 108, 109])
    #
    # --- 测试批次数据 ---
    # 批次 0:
    #   输入形状: torch.Size([2, 4])
    #   输入数据:
    # tensor([[101, 102, 103, 104],
    #         [103, 104, 105, 106]])
    #   目标形状: torch.Size([2, 4])
    #   目标数据:
    # tensor([[102, 103, 104, 105],
    #         [104, 105, 106, 107]])
    # 批次 1:
    #   输入形状: torch.Size([2, 4])
    #   输入数据:
    # tensor([[105, 106, 107, 108],
    #         [107, 108, 109, 110]])
    #   目标形状: torch.Size([2, 4])
    #   目标数据:
    # tensor([[106, 107, 108, 109],
    #         [108, 109, 110,   0]])
    #
    # --- 测试不同步长 ---
    # 步长为 1 时的样本数: 7
    #   第一个样本输入: tensor([101, 102, 103, 104])
    #   第一个样本目标: tensor([102, 103, 104, 105])
    #   第二个样本输入: tensor([102, 103, 104, 105])
    #   第二个样本目标: tensor([103, 104, 105, 106])
    # 步长为 2 时的样本数: 4
    #   第一个样本输入: tensor([101, 102, 103, 104])
    #   第一个样本目标: tensor([102, 103, 104, 105])
    #   第二个样本输入: tensor([103, 104, 105, 106])
    #   第二个样本目标: tensor([104, 105, 106, 107])
    # 步长为 3 时的样本数: 3
    #   第一个样本输入: tensor([101, 102, 103, 104])
    #   第一个样本目标: tensor([102, 103, 104, 105])
    #   第二个样本输入: tensor([104, 105, 106, 107])
    #   第二个样本目标: tensor([105, 106, 107, 108])