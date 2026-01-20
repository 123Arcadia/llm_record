import json
import os.path
import re
from functools import partial

import requests
import tiktoken
import time
import torch
import torch.nn as nn
from pycparser.ply.yacc import token
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from gpt_download import download_and_load_gpt2
from previous_chapters import GPTModel, load_weights_into_gpt, calc_loss_loader, train_model_simple, plot_losses
from previous_chapters import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

# 加载数据
file_path = "instruction-data.json"
url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/01_main-chapter-code/instruction-data.json"
)



class IntstructionDataSet(Dataset):
    def __init__(self, data, tokenizer):
        super().__init__()
        assert len(data) > 0
        # pre-tokenizer
        self.data = data
        self.encodeed_texts = []
        for e in data:
            inst_plus_input = format_input(e)
            res_text = f"\n\n### Response:\n{e['output']}"

            # inst_plus_input = format_input_Phi_3(entry)
            # res_text = f"\n<|assistant|>:\n{entry['output']}"

            full_text = inst_plus_input + res_text
            self.encodeed_texts.append(tokenizer.encode(full_text))

    def __getitem__(self, idx):
        return self.encodeed_texts[idx]

    def __len__(self):
        return len(self.data)

def download_and_load_file(file_path, url):
    try:
        if not os.path.exists(file_path):
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            if res.status_code == 200:
                print(f'返回成功')
            else:
                print(f'请求失败')
            text  = res.text
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(f'{file_path}已存在')
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f'下载失败!{e=}')
        return []
    return data

def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text

def format_input_Phi_3():
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )
    input_text = f"\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text


def formatinput(data):
    model_input = format_input(data[0])
    desired_response = f"\n\n### Response:\n{data[0]['output']}"

    print(model_input + desired_response)

def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
    """
    建立input_batch
    :param batch:
    :param pad_token_id:
    :param device:
    :return:
    """
    bsz_max_len = max(len(t)+1 for t in batch)
    inputs_lst = []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (new_item + [pad_token_id] * (bsz_max_len - len(new_item)))
        inputs = torch.tensor(padded[:-1])
        inputs_lst.append(inputs)
    o = torch.stack(inputs_lst).to(device)
    return o


def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
    """
    建立target_batch (相对于input_batch后移一位token)
    """
    bsz_max_len = max(len(t)+1 for t in batch)
    input_lst, target_lst = [], []
    for item in batch:
        new = item.copy()
        new += [pad_token_id]
        padded = (new + [pad_token_id] * (bsz_max_len - len(new)))
        input_lst.append(torch.tensor(padded[:-1]))
        target_lst.append(torch.tensor(padded[1:]))

    input_batch = torch.stack(input_lst).to(device)
    target_batch = torch.stack(target_lst).to(device)
    return input_batch, target_batch

def custom_collate_draft_fn(batch, pad_token_id=50256, ingore_index=-100, allowed_max_length=None, device="cpu"):
    """
    添加ignore_index: 对于pad_token后期loss略过
    allowed_max_len: 控制上下文长度
    """
    bsz_max_len = max(len(t)+1 for t in batch)
    input_lst, target_lst = [], []

    for item in batch:
        new = item.copy()
        new += [pad_token_id]
        padded = (new + [pad_token_id] * (bsz_max_len - len(new)))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # 把除了第一个pad_token_id以外的pad换成-100
        mask = targets == pad_token_id
        idxs = torch.nonzero(mask).squeeze()
        if idxs.numel() > 1:
            targets[idxs[1:]] = ingore_index # 略过第一个pad_token
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        input_lst.append(inputs)
        target_lst.append(targets)

    input_batch = torch.stack(input_lst).to(device)
    target_batch = torch.stack(target_lst).to(device)
    return input_batch, target_batch


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = download_and_load_file(file_path, url)
    # data = []
    # print("Example entry:\n", data[50])
    # print(f'{len(data)}') # 1100

    # formatinput(data) # 测试函数
    # ### Instruction:
    # Identify the correct spelling of the following word.
    #
    # ### Input:
    # Ocassion
    #
    # ### Response:
    # The correct spelling is 'Occasion.'

    print(f'{len(data)=}')

    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)
    val_portion = len(data) - train_portion - test_portion
    train_data = data[:train_portion]
    val_data = data[train_portion:train_portion + val_portion]
    test_data = data[train_portion + val_portion:]

    # 把|<endoftext>|作为pad_token
    tokenizer = tiktoken.get_encoding("gpt2")

    customized_collate_fn = partial(
        custom_collate_draft_fn,
        device=device,
        allowed_max_length=1024
    )

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = IntstructionDataSet(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = IntstructionDataSet(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = IntstructionDataSet(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # print("Train loader:")
    # for inputs, targets in train_loader:
    #     print(inputs.shape, targets.shape)
        # torch.Size([8, 61]) torch.Size([8, 61])
        # ...
        # torch.Size([8, 69]) torch.Size([8, 69])

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

    CHOOSE_MODEL = "gpt2-small (124M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    torch.manual_seed(123)
    input_text = format_input(val_data[0])
    # format_input_Phi_3
    print(f'input_text: \n{input_text}')
    token_ids = generate(model=model, idx=text_to_token_ids(input_text, tokenizer),
                         max_new_tokens=35, context_size=BASE_CONFIG["context_length"], eos_id=50256)
    generated_text = token_ids_to_text(token_ids, tokenizer)
    print(f'Generated text: \n{generated_text}')

    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )
    # response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
    print(f'response_text: \n{response_text}')


    print()
    print('='*50)
    print()
    print(f'训练前损失验证:')
    model.to(device)

    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    print()
    print('='*50)
    print(f'训练')
    print()
    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
                                        model, train_loader, val_loader, optimizer, device,
                                        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
                                        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


    print()
    print('=' * 50)
    print("训练完成再次验证模型能力")
    print()
    torch.manual_seed(123)

    for entry in test_data[:3]:
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        print(f'{input_text=}')
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

    print()
    print('='*50)
    print("保存模型响应")
    print()
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        test_data[i]["model_response"] = response_text

    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

    print()
    print('='*50)
    print("保存模型")
    print()
    # 移除字符串 CHOOSE_MODEL 中所有的空格、左括号 (、右括号 )
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL)}-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

    # Load model via
    # model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))


    # > python exercise_experiments.py --exercise_solution phi3_prompt

