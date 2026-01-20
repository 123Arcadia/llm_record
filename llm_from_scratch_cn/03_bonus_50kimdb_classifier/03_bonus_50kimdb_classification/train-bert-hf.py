import argparse
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer



class IMDBDataset(Dataset):
    def __init__(self, csv_file, max_length=None, tokenizer=None, pad_token_id = 50526):
        super().__init__()
        self.data = pd.read_csv(csv_file)
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        self.encoded_texts = [
            tokenizer.encode(txt)[:max_length] for txt in self.data['text']
        ]
        pad_token_id = 0
        self.encoded_texts= [
            et + [pad_token_id]*(max_length-len(et))
            for et in self.encoded_texts
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        encode = self.encoded_texts[i]
        label = self.data.iloc[i]['label']
        return torch.tensor(encode, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def _longest_encoded_length(self, tokenizer):
        max_l = 0
        for txt in self.data['text']:
            encoded_l = len(tokenizer.encode(txt))
            if encoded_l > max_l:
                max_l = encoded_l
        return max_l


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # logits = model(input_batch)[:, -1, :]  # Logits of last output token
    logits = model(input_batch).logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


def calc_loss_loader(data_loder, model, device, num_batches=None):
    total_loss = 0
    if num_batches is None:
        num_batches = len(data_loder)
    else:
        num_batches = min(num_batches, len(data_loder))
    for i, (input_batch, target_batch) in enumerate(data_loder):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device,)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()
def calc_acc_loader(data_loader, model, device, num_batches=None):
    # 平均每个batch的正确率
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            logits = model(input_batch).logits
            preicted_labels = torch.argmax(logits, dim=-1)
            num_examples += preicted_labels.shape[0]
            correct_predictions += (preicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
    model.train()
    return train_loss, val_loss



def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epoches, eval_freq, eval_iter,
                            tokenizer, max_step=None):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    for epoch in range(num_epoches):
        model.train()
        for input_bacth, traget_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_bacth, traget_batch, model, device)
            loss.backward()
            optimizer.step()
            examples_seen += input_bacth.shape[0]
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f'Epoch {epoch+1} (Step {global_step:06d}:'
                      f'Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}')

            if max_step is not None and global_step > max_step:
                break
    return train_losses, val_losses, train_accs, val_accs, examples_seen




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="bert-hf")
    parser.add_argument("--trainable_layers", type=str, default="last_block",
                        help="Which layers to train. Options: 'all', 'last_block', 'last_layer'.")
    parser.add_argument(
        "--bert_model",
        type=str,
        default="distilbert",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    args = parser.parse_args()
    ### 加载模型
    torch.manual_seed(123)
    if args.bert_model == 'distilbert':
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        model.out_head = torch.nn.Linear(in_features=768, out_features=2)
        if args.trainable_layers == 'last_layer':
            pass
        elif args.trainable_layers == 'last_block':
            for p in model.pre_classifier.parameters():
                p.requires_grad = True
            for param in model.distilbert.transformer.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for p in model.parameters():
                p.requires_grad = True
        else:
            raise ValueError("Invalid -- trainable_layers argument")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    elif args.bert_model == 'roberta':
        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-large", num_labels=2
        )
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)

        if args.trainable_layers == "last_layer":
            pass
        elif args.trainable_layers == "last_block":
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.roberta.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")

    else:
        raise ValueError("Selected --bert_model not Supported")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    pad_token_id = tokenizer.encode(tokenizer.pad_token)
    base_path = Path("")
    train_dataset = IMDBDataset(base_path / "train.csv", max_length=256, tokenizer=tokenizer, pad_token_id=pad_token_id)
    val_dataset = IMDBDataset(base_path / "validation.csv", max_length=256, tokenizer=tokenizer,
                              pad_token_id=pad_token_id)
    test_dataset = IMDBDataset(base_path / "test.csv", max_length=256, tokenizer=tokenizer, pad_token_id=pad_token_id)

    num_workers = 0
    batch_size = 8

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    st = time.time()
    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epoches = 3
    train_losses, val_losses, trains_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, num_epoches=num_epoches, eval_freq=50, eval_iter=20,
        tokenizer=tokenizer, max_step = None
    )

    et = time.time()
    execution_time_minutes = (et - st) / 60
    print(f"Training completed in {execution_time_minutes} min")

    print('\nEvaluationg on the full datasets ..\n')
    train_acc = calc_acc_loader(train_loader, model, device)
    val_acc = calc_acc_loader(val_loader, model, device)
    test_acc = calc_acc_loader(test_loader, model, device)

    print(f"Training accuracy: {train_acc * 100:.2f}%")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
















