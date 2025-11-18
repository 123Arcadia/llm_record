"""
从公开的OpenAI加载权重

"""

import torch
import tiktoken
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from llm_from_scratch_cn.chapter5.exam05_train.main_code_chapter01.previous_chapter import create_dataloader_v1, generate
from previous_chapter import GPTModel, generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<endoftext>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加batch维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


GPT_CONFIG_124M = {
    "vocab_size": 50257,  # Vocabulary size
    "context_length": 256,  # Shortened context length (orig: 1024)
    "emb_dim": 768,  # Embedding dimension
    "n_heads": 12,  # Number of attention heads
    "n_layers": 12,  # Number of layers
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False  # Query-key-value bias
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()  # Disable dropout during inference

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

##################################
# 读取文本the-verdict.txt
##################################
file_path = "../../../the-verdict.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()

total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)
# Characters: 20479
# Tokens: 5145


##################################
# 划分训练集/验证集
##################################
train_ratio = 0.90
split_text = int(train_ratio * len(text_data))
train_data = text_data[:split_text]
val_data = text_data[split_text:]
torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'], drop_last=True, shuffle=True, num_workers=0
)
val_loader = create_dataloader_v1(
    val_data, batch_size=2, max_length=GPT_CONFIG_124M['context_length'],
    stride=GPT_CONFIG_124M['context_length'], drop_last=False, shuffle=False, num_workers=0
)
if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

# for x, y in train_loader:
#     print(x.shape, y.shape)
#     # torch.Size([2, 256]) torch.Size([2, 256])

train_tokens = 0
for input_batch, target_batch in train_loader:
    train_tokens += input_batch.numel()

val_tokens = 0
for input_batch, target_batch in val_loader:
    val_tokens += input_batch.numel()

print("Training tokens:", train_tokens)
print("Validation tokens:", val_tokens)
print("All tokens:", train_tokens + val_tokens)
# Training tokens: 4608
# Validation tokens: 512
# All tokens: 5120


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算给定batch的交叉熵
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    在指定数据加载器中数据集的计算loss
    """
    totaL_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            totaL_loss += loss
        else:
            break
    return totaL_loss / num_batches


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device.")
model.to(device)
torch.manual_seed(123)
with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)
print(f'{train_loss.item()=:,}')
print(f'{val_loss.item()=:,}')
# train_loss.item()=10.987584114074707
# val_loss.item()=10.98110580444336


######################################
# 训练模型
######################################
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size, )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epoches, eval_freq, eval_iter,
                       start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epoches):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f'Ep {epoch} {global_step:06d}: '
                      f'Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}')

        # 在每个epoch后输出采样文本
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen



torch.manual_seed(123)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device, num_epoches=num_epochs,
    eval_freq=5, eval_iter=5, start_context="Every effort moves you", tokenizer=tokenizer
)
print(f'{len(train_losses)=}, {len(val_losses)=}, {len(tokens_seen)=}')

####################################
# 保存模型、参数
####################################

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    },
    "model_and_optimizer.pth"
)


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Training Loss")
    ax1.plot(epochs_seen, val_losses, label="Validation Loss", linestyle="-.")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis

    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()  # Adjust layout to make room
    plt.savefig("./train_output/loss-plot.jpg", dpi=300)
    plt.show()


# 在[0, num_epochs]平均分为len(train_losses)个元素
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses), device='cpu')
# print(f'{tokens_seen}')
# print(f'{train_losses}')
# print(f'{val_losses}')

train_losses = [x.to('cpu') for x in train_losses]
val_losses = [x.to('cpu') for x in val_losses]
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)



## 结果:
# Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I
# len(train_losses)=18, len(val_losses)=18, len(tokens_seen)=18




##########################################
# 使用topk、temperature推理
##########################################
torch.manual_seed(123)
inference_device = 'cuda'
token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(inference_device),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4
)

print("inference Output text:\n", token_ids_to_text(token_ids, tokenizer))