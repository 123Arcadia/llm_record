import os

import zipfile
from pathlib import Path

import pandas as pd
import requests
import tiktoken
import torch
import time

from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset, DataLoader
from gpt_download import download_and_load_gpt2
from previous_chapter import GPTModel, load_weights_into_gpt
from previous_chapter import (
    generate_text_simple,
    text_to_token_ids,
    token_ids_to_text
)


url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):
    if data_file_path.exists():
        print(f'{data_file_path=} exists. Skipping download and extraction.')
        return
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                out_file.write(chunk)

    # 解压
    with zipfile.ZipFile(zip_path) as zip_file:
        zip_file.extractall(extracted_path)

    ori_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(ori_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path=}")

def down_data(url):
    try:
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)
    except (requests.exceptions.RequestException, TimeoutError) as e:
        print(f"Primary URL failed: {e}. Trying backup URL...")
        url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip"
        download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    # print(f'{num_spam=}') #747
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df, train_frac, validation_frac):
    # Shuffle the entire DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calculate split indices
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Split the DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        if max_length is None:
            self.max_length = max([len(text) for text in self.encoded_texts])
        else:
            self.max_length = max_length
            self.encoded_texts = [t[:max_length] for t in self.encoded_texts]

        self.encoded_texts = [t + [pad_token_id] * (self.max_length - len(t)) for t in self.encoded_texts]

    def __getitem__(self, i):
        return (
            torch.tensor(self.encoded_texts[i], dtype=torch.long),
            torch.tensor(self.data.iloc[i]["Label"], dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = max(num_batches, len(data_loader))

    for i , (in_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            in_batch, target_batch = in_batch.to(device), target_batch.to(device)
            with torch.no_grad():
                logits = model(in_batch)[:, -1, :]
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss

# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, valid_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        valid_loss = calc_loss_loader(valid_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, valid_loss

def train_classifier_simple(model, train_loader, valid_loader, optimizer, device, num_epoches, eval_freq, eval_iter):

    train_losses, valid_losses, train_accs, valid_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    for epoch in range(num_epoches):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            examples_seen += input_batch.shap[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, valid_loss = evaluate_model(model, train_loader, valid_loader, eval_iter)
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                print(f"Ep {epoch + 1} (Step {global_step:06d}): Train loss {train_loss:.3f}, Val loss {valid_loss:.3f}")

        #计算acc
        train_acc = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        valid_acc = calc_accuracy_loader(valid_loader, model, device, num_batches=eval_iter)
        print(f'Train acc: {train_acc*100:.2f}%')
        print(f'Valid acc: {valid_acc*100:.2f}%')
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
    return train_losses, val_losses, train_accs, valid_accs, examples_seen

def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    # Create a second x-axis for examples seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(examples_seen, train_values, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()  # Adjust layout to make room
    plt.savefig(f"{label}-plot.jpg", dpi=300)
    # plt.show()


def save_model(model, file_name, optimizer):
    model.eval()
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, file_name)
    print(f"Model saved to {file_name}")
    model.train()


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    # Prepare inputs to the model
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]
    # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
    # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

    # Truncate sequences if they too long
    input_ids = input_ids[:min(max_length, supported_context_length)]
    assert max_length is not None, (
        "max_length must be specified. If you want to use the full model context, "
        "pass max_length=model.pos_emb.weight.shape[0]."
    )
    assert max_length <= supported_context_length, (
        f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
    )
    # Alternatively, a more robust version is the following one, which handles the max_length=None case better
    # max_len = min(max_length,supported_context_length) if max_length else supported_context_length
    # input_ids = input_ids[:max_len]

    # Pad sequences to the longest sequence
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)  # add batch dimension

    # Model inference
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]  # Logits of the last output token
    predicted_label = torch.argmax(logits, dim=-1).item()

    # Return the classified result
    return "spam" if predicted_label == 1 else "not spam"


if __name__ == '__main__':
    """
    # down_data(url)
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
    print(df["Label"].value_counts())
    # Label
    # ham     4825
    # spam     747
    # Name: count, dtype: int64
    print(df["Label"].shape) # (5572,)

    # 样本平衡
    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # Test size is implied to be 0.2 as the remainder

    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)
    """

    tokenizer = tiktoken.get_encoding("gpt2")
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    train_path = "./sms_spam_collection/train.csv"
    valid_path = "./sms_spam_collection/validation.csv"
    test_path = "./sms_spam_collection/test.csv"
    train_dataset = SpamDataset(csv_file=train_path, max_length=None, tokenizer=tokenizer)
    valid_dataset = SpamDataset(csv_file=valid_path, max_length=train_dataset.max_length, tokenizer=tokenizer)
    test_dataset = SpamDataset(csv_file=test_path, max_length=train_dataset.max_length, tokenizer=tokenizer)

    # 建立dataloader
    num_workers = 0
    bsz = 8
    torch.manual_seed(123)
    train_loader = DataLoader(train_dataset, batch_size=bsz, num_workers=num_workers, drop_last=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=bsz, num_workers=num_workers, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=bsz, num_workers=num_workers, drop_last=False)

    # for text, label in train_loader:
    #     print(f'{text.shape=} {label=}')
    # text.shape=torch.Size([8, 120]) label=tensor([1, 1, 0, 0, 0, 0, 1, 0])
    # text.shape=torch.Size([8, 120]) label=tensor([1, 0, 1, 0, 0, 0, 0, 0])
    # print(f'{len(train_loader)}') # 130
    # print(f'{len(valid_loader)}') # 19
    # print(f'{len(test_loader)}') # 38

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    text_1 = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )
    print('==================事前验证======================')
    print(token_ids_to_text(token_ids, tokenizer))
    # 注意: As we can see, the model is not very good at following instructions
    # This is expected, since it has only been pretrained and not instruction-finetuned (instruction finetuning will be covered in the next chapter)

    ##########################the output layer#############################
    # The goal is to replace and finetune the output layer
    # 冻结 model
    for p in model.parameters():
        p.requires_grad = False
    # 要把model.out_head的参数设置重新设置为二分类
    num_classes = 2
    model.out_head = nn.Linear(BASE_CONFIG["emb_dim"], num_classes)
    # 研究表明和最后一个block一起训练效果更好
    for p in model.trf_blocks[-1].parameters():
        p.requires_grad = True
    for p in model.final_norm.parameters():
        p.requires_grad = True

    # 按照attn mask机制只有last token（即outs[:,-1,:]）包含了所有的token信息
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    train_acc = calc_accuracy_loader(train_loader, model, device, num_examples=10)
    valid_acc = calc_accuracy_loader(valid_loader, model, device, num_examples=10)
    test_acc = calc_accuracy_loader(test_loader, model, device, num_examples=10)
    print()
    print(f'==================微调前的准确率===================')
    print(f'Train_acc: {train_acc*100:.2f}%')
    print(f'Valid_acc: {valid_acc*100:.2f}%')
    print(f'Test_acc: {test_acc*100:.2f}%')
    print()
    print(f'==================(设计新的loss calc后)的准确率===================')
    with torch.no_grad():  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(valid_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")

    s = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epoches = 5
    train_losses, val_losses, train_accs, valid_accs, examples_seen = train_classifier_simple(model, train_loader, valid_loader, optimizer, device, num_epoches,
                                                       eval_freq=50, eval_iter=5)
    e = time.time()
    times = (e - s) / 60
    print(f"Training time: {times:.2f} minutes")
    # 保存模型
    save_model(model, "spam_classifier.pt", optimizer)

    # 画图
    epochs_tensor = torch.linspace(0, num_epoches, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    # loss 图
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
    # 准确率
    epochs_tensor = torch.linspace(0, num_epoches, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(epochs_tensor, examples_seen_tensor, train_accs, valid_accs, label="accuracy")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(valid_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print()
    print("==========================微调后==========================")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")


    print("==========================use model==========================")
    text_1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )
    print(f'Text1: {text_1}')

    print("Result:", classify_review(
        text_1, model, tokenizer, device, max_length=train_dataset.max_length
    ))
    text_2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )
    print(f'Text2: {text_1}')

    print("Result:", classify_review(
        text_2, model, tokenizer, device, max_length=train_dataset.max_length
    ))


