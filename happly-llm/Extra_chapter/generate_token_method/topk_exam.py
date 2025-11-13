import torch
import torch.nn.functional as F


def _random_sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
    """
    随机采样：基于概率分布随机选择token

    Args:
        logits: 模型输出的logits，形状为 (batch_size, vocab_size)
        temperature: 温度参数，控制随机性
        top_k: 只考虑概率最高的k个token

    Returns:
        选择的token索引，形状为 (batch_size, 1)
    """
    # 1. 温度缩放
    logits = logits / temperature

    # 2. Top-k过滤
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

    # 3. 计算概率并采样
    probs = F.softmax(logits, dim=-1)
    # 按照多箱随机变量概率分布返回索引
    idx_next = torch.multinomial(probs, num_samples=1)
    return idx_next