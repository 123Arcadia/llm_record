# 加载定义好的模型参数-此处以 Qwen-2.5-1.5B 为例
# 使用 transforemrs 的 Config 类进行加载
from torchdata.nodes import IterableWrapper
# from torchdata.datapipes.iter import IterableWrapper
from transformers import AutoConfig
import torch

# 加载一个预训练好的 tokenizer
from transformers import AutoTokenizer

# 使用该配置生成一个定义好的模型
from transformers import AutoModelForCausalLM

# 加载预训练数据
from datasets import load_dataset
# 预训练一般将文本拼接成固定长度的文本段
from itertools import chain

from transformers import TrainingArguments
# 配置训练参数
from transformers import Trainer, default_data_collator





# model.to(device)
# # print(f'{model=}')
# n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())

model_path = "./autodl-tmp/qwen-1.5b"
text = "你好, 我是一名学生"
tokenizer = AutoTokenizer.from_pretrained(model_path)
#
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = AutoConfig.from_pretrained(model_path)
# print(f'{config=}')
model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)


# tok = tokenizer(text)
# print(f'{tok=}')
# # tok={'input_ids': [108386, 11, 49434, 239, 110124, 99720], 'attention_mask': [1, 1, 1, 1, 1, 1]}
# out_encode = tokenizer.encode(text, add_special_tokens=True)
# print(f'{out_encode=}')
# # out_encode=[108386, 11, 49434, 239, 110124, 99720]
#
# out_decode = tokenizer.decode(out_encode)
# print(f'{out_decode=}')
# # out_decode='你好, 我是一名学生'
