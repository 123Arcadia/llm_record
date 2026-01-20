
# add ___special tokens___ to a tokenizer

import tiktoken
import torch.nn
from previous_chapter import GPTModel


if __name__ == '__main__':

    base_tokenizer = tiktoken.get_encoding("gpt2")
    sample_text = "Hello, MyNewToken_1 is a new token. <|endoftext|>"

    # allowed_special 参数指定允许的特殊令牌，避免编码时抛出异常
    token_ids = base_tokenizer.encode(sample_text, allowed_special={"<|endoftext|>"})

    for token_id in token_ids:
        print(token_id, " -> ", base_tokenizer.decode([token_id]))
        # 15496  ->  Hello
        # 11  ->  ,
        # 2011  ->   My
        # 3791  ->  New
        # 30642  ->  Token
        # 62  ->  _
        # 16  ->  1
        # 318  ->   is
        # 257  ->   a
        # 649  ->   new
        # 11241  ->   token
        # 13  ->  .
        # 220  ->
        # 50256  ->  <|endoftext|>

    """
    默认是把MyNewToken_1单词分开，但我们希望把它当做一个整体来标记
    """

    # Define custom tokens and their token IDs
    custom_tokens = ["MyNewToken_1", "MyNewToken_2"]
    custom_token_ids = {
        token: base_tokenizer.n_vocab + i for i, token in enumerate(custom_tokens)
    }

    # print(base_tokenizer.n_vocab)
    # # 50257
    # print(base_tokenizer.eot_token)
    # # 50256
    # print(base_tokenizer.name)
    # # gpt2
    # print(base_tokenizer.max_token_value)
    # # 50256
    # # print(base_tokenizer.__dict__)
    # print(base_tokenizer.special_tokens_set)
    # # {'<|endoftext|>'}
    print("##########################################")

    # Create a new Encoding object with extended tokens
    extended_tokenizer = tiktoken.Encoding(
        name="gpt2_custom",
        pat_str=base_tokenizer._pat_str,
        mergeable_ranks=base_tokenizer._mergeable_ranks,
        special_tokens={**base_tokenizer._special_tokens, **custom_token_ids},
    )

    special_tokens_set = set(custom_tokens) | {"<|endoftext|>"}

    token_ids = extended_tokenizer.encode(
        "Sample text with MyNewToken_1 and MyNewToken_2. <|endoftext|>",
        allowed_special=special_tokens_set
    )
    print(token_ids)
    # [36674, 2420, 351, 220, 50257, 290, 220, 50258, 13, 220, 50256]
    for token_id in token_ids:
        print(token_id, " -> ", extended_tokenizer.decode([token_id]))
        # 36674  ->  Sample
        # 2420  ->   text
        # 351  ->   with
        # 220  ->
        # 50257  ->  MyNewToken_1
        # 290  ->   and
        # 220  ->
        # 50258  ->  MyNewToken_2
        # 13  ->  .
        # 220  ->
        # 50256  ->  <|endoftext|>



    ###########################################################
    # 看到，MyNewToken_1和MyNewToken_2被编码成两个新的token_id，并且被标记为特殊token
    # update the embedding and output layers of the LLM
    ##################### 更新 Embedding层 ###########################


    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False  # Query-Key-Value bias
    }

    model = GPTModel(GPT_CONFIG_124M)
    # 更新模型中对应的Embedding层
    num_tokens, emb_size = model.tok_emb.weight.shape
    new_num_tokens = num_tokens + 2 # 我们新加了两个特殊标记s
    # 创建新Embedding
    new_emb = torch.nn.Embedding(new_num_tokens, emb_size)
    new_emb.weight.data[:num_tokens] = model.tok_emb.weight.data # 前num_tokens的权重是原有的不变
    # 更新model中的tok_emb
    model.tok_emb = new_emb
    print(f'{model.tok_emb=}')
    # model.tok_emb=Embedding(50259, 768)
    # 可以看到，更新Embedding成功！

    ##################### 更新 out_head层(无权重绑定时) ###########################

    original_out_features, original_in_features = model.out_head.weight.shape

    # Define the new number of output features (e.g., adding 2 new tokens)
    new_out_features = original_out_features + 2

    # Create a new linear layer with the extended output size
    new_linear = torch.nn.Linear(original_in_features, new_out_features)

    # Copy the weights and biases from the original linear layer
    # 更新weight和bias
    with torch.no_grad():
        new_linear.weight[:original_out_features] = model.out_head.weight
        if model.out_head.bias is not None:
            new_linear.bias[:original_out_features] = model.out_head.bias

    # Replace the original linear layer with the new one
    model.out_head = new_linear

    print(f'{model.out_head=}')
    # model.out_head=Linear(in_features=768, out_features=50259, bias=True)

    ##################### 更新 out_head层(有权重绑定时) ###########################
    # 直接复制embedding的weight给out_head层
    model.out_head.weight = model.tok_emb.weight.data
















