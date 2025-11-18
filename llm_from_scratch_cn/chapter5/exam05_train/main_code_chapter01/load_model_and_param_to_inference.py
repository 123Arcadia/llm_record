import tiktoken
import torch

from llm_from_scratch_cn.chapter5.exam05_train.main_code_chapter01.previous_chapter import GPTModel, generate

GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-key-value bias
}

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<endoftext>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加batch维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())

def main():
    device = 'cuda' if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.eval()
    # print(f'{checkpoint=}')
    # print(f'{model=}')

    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to('cuda'),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4
    )

    print("[inference]Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == '__main__':
    main()
    # [inference]Output text:
    #  Every effort moves you say terr the to do the axi," he that with a of.