
from importlib.metadata import version
import torch
import torch.nn.functional as F
from transformers.models.tapas.modeling_tapas import flatten

pkgs = ["matplotlib", "numpy", "tiktoken", "torch"]
for p in pkgs:
    print(f"{p} version: {version(p)}")

"""
计算文本困惑度和交叉熵
"""
class GPTModel:
    # todo
    pass

inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
                       [40,    1107, 588]])   #  "I really like"]

targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
                        [588,  428,  11311]]) #  " really like chocolate"]


model = GPTModel()
with torch.no_grad():
    logits = model(inputs)

probs = F.softmax(logits, dim=-1)
print(f'{probs.shape=}')
# [bs, num_tokens, (seq_len)vocab_size] (num_tokens = heads * seq_len)


# greedy
token_ids = torch.argmax(probs, dim=-1, keepdim=True)
print(f'{token_ids=}')



# 计算交叉熵
cross_entropy_loss = F.cross_entropy(logits,flatten(0, 1), targets.flatten())
# 计算困惑度
perplexity = torch.exp(cross_entropy_loss)

# 计算所有batch的交叉熵
total_loss = 0.0
batch = 8
total_loss += cross_entropy_loss.item()
# 最后
total_loss /= total_loss / batch











