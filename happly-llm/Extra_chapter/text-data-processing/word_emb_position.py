import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
# 定义嵌入参数
vocab_size = 50257 # gpt-2的词汇表
output_dim = 256
context_length= 4 # 上下文长度(即输入的最大序列长度)

# 创建词嵌入层和位置嵌入层
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# 生成位置索引
pos_indices= torch.arange(context_length)
pos_embeddings = pos_embedding_layer(pos_indices) # [4, 256]

def test_position_emb():
    input_ids = torch.tensor([
        [40, 367, 2885, 1464],  # 第一句的词ID
        [1807, 3619, 402, 271],  # 第二句的词ID
        [10899, 2138, 257, 7026]  # 第三句的词ID
    ])

    batch_size, seq_len = input_ids.shape
    print(f'{input_ids.shape=}')

    # 生成词嵌入
    token_emb = token_embedding_layer(input_ids)
    print(f"词嵌入: {token_emb.shape=}")

    # 添加位置嵌入
    input_emb = token_emb + pos_embeddings
    print(f'添加位置嵌入: {input_emb.shape=}')

    # 验证位置嵌入的唯一性
    print("\n位置嵌入向量（前3个位置的前5维）:")

    for i in range(3):
        print(f'位置:{i} {pos_embeddings[i, :5]}')
    # input_ids.shape=torch.Size([3, 4])
    # 词嵌入: token_emb.shape=torch.Size([3, 4, 256])
    # 添加位置嵌入: input_emb.shape=torch.Size([3, 4, 256])
    #
    # 位置嵌入向量（前3个位置的前5维）:
    # 位置:0 tensor([ 1.1364,  1.1264, -0.3382,  0.6830, -1.2337], grad_fn=<SliceBackward0>)
    # 位置:1 tensor([-0.1462,  0.6811,  0.1754, -0.0835,  2.1103], grad_fn=<SliceBackward0>)
    # 位置:2 tensor([-0.7912,  1.1004, -0.6810,  1.0889, -0.4031], grad_fn=<SliceBackward0>)