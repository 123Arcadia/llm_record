import argparse
import os
import time

import torch
from torch import nn

import data
import math
from model import TransformerModel, RNNModel, PositionalEncoding


parser = argparse.ArgumentParser(description="PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model")

parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings') # max_length
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument('--dry-run', action='store_true',
                    help='verify the code and the model')
parser.add_argument('--accel', action='store_true',
                    help='Enables accelerated training')
parser.add_argument('--use-optimizer', action='store_true',
                    help='Uses AdamW optimizer for gradient updating')
args = parser.parse_args()

torch.manual_seed(args.seed)
# if args.accel and torch.accelerator.is_available():
#     device = torch.accelerator.current_accelerator()
#
# else:
#     device = torch.device("cpu")

if args.accel:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'
device = torch.device(device)
print("Using device:", device)


print("####################")
print('load data')
print("####################")


corpus = data.Corpus(args.data)
print(f"{len(corpus.dictionary)=}")
print(f"{corpus.train.shape=}")
print(f"{corpus.valid.shape=}")
print(f"{corpus.test.shape=}")
print(f'{len(corpus.dictionary)=}')
print(f'{len(corpus.dictionary.word2idx)=}')
# len(corpus.dictionary)=18328
# corpus.train.shape=torch.Size([217645])
# corpus.valid.shape=torch.Size([217645])
# corpus.test.shape=torch.Size([245568])
# len(corpus.dictionary)=18328
# len(corpus.dictionary.word2idx)=18328
eval_batch_size = 10


def bacthify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz) # 去掉(无法除尽)多余的元素
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


train_data = bacthify(corpus.train, args.batch_size)
valid_data = bacthify(corpus.train, eval_batch_size)
test_data  = bacthify(corpus.train, eval_batch_size)
print(f'{train_data.size()=}')
# print(f'{train_data.size()=}')
# print(f'{valid_data.size()=}')
# print(f'{test_data.size()=}')
# # train_data.size()=torch.Size([10882, 20])
# # valid_data.size()=torch.Size([21764, 10])
# # test_data.size()=torch.Size([21764, 10])

print("####################")
print('build model')
print("####################")
print(f'{args=}')
# args=Namespace(data='./data/wikitext-2', model='LSTM', emsize=200, nhid=200, nlayers=2,
# lr=20, clip=0.25, epochs=6, batch_size=20, bptt=35, dropout=0.2, tied=False, seed=1111, log_interval=200,
# save='model.pt', onnx_export='', nhead=2, dry_run=False, accel=True, use_optimizer=False)
ntokens = len(corpus.dictionary)
if args.model == 'Transformer':
    model = TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
else:
    model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

criterion = nn.NLLLoss() # 模型已经已经log_softmax

if args.use_optimizer:
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source)-1-i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target

def evaluate(data_source):
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0)-1, args.bptt):
            data, targets = get_batch(data_source, i)
            # print(f'{i=} | {data.shape=}')
            # LSTM: i=3500 | data.shape=torch.Size([35, 10])
            if args.model == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)

            else:
                output, hidden = model(data, hidden)
                print(f'{output.shape=}')
                # output.shape=torch.Size([350, 18328])
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
        return total_loss / (len(data_source) - 1)

def train():
    model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(args.batch_size)

    for batch, i in enumerate(range(0, train_data.size(0)-1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # print(f"11 {data.shape=}") # data.shape=torch.Size([35, 20])

        if args.use_optimizer:
            optimizer.zero_grad()
        else:
            model.zero_grad()
        if args.model == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

        loss = criterion(output, targets)
        loss.backward()

        # clip_grad_norm有助于防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if args.use_optimizer:
            optimizer.step()
        else:
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                              elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if args.dry_run: # ??? false
            break


def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


lr = args.lr
best_val_loss = None
train_times = []
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()

        train()

        val_loss = evaluate(valid_data)
        print('-' * 89)
        epoch_et = time.time()
        train_times.append(epoch_et - epoch_start_time)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (epoch_et - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = val_loss
        else:
            # 在valid_data上无改进，就降低lr
            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

print(f'Train times: {sum(train_times):.2f}')
# Load the best saved model.
with open(args.save, 'rb') as f:
    if args.model == 'Transformer':
        safe_globals = [
            PositionalEncoding,
            TransformerModel,
            torch.nn.functional.relu,
            torch.nn.modules.activation.MultiheadAttention,
            torch.nn.modules.container.ModuleList,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.linear.NonDynamicallyQuantizableLinear,
            torch.nn.modules.normalization.LayerNorm,
            torch.nn.modules.sparse.Embedding,
            torch.nn.modules.transformer.TransformerEncoder,
            torch.nn.modules.transformer.TransformerEncoderLayer,
        ]
    else:
        safe_globals = [
            RNNModel,
            torch.nn.modules.dropout.Dropout,
            torch.nn.modules.linear.Linear,
            torch.nn.modules.rnn.GRU,
            torch.nn.modules.rnn.LSTM,
            torch.nn.modules.rnn.RNN,
            torch.nn.modules.sparse.Embedding,
        ]
    with torch.serialization.safe_globals(safe_globals):
        model = torch.load(f)
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    # Currently, only rnn model supports flatten_parameters function.
    if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
        model.rnn.flatten_parameters()


# Run on test data.

test_st = time.time()
test_loss = evaluate(test_data)
test_et = time.time()
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f} times: {:.2f}'.format(
    test_loss, math.exp(test_loss), test_et-test_st))
print('=' * 89)

if len(args.onnx_export) > 0:
    # Export the model in ONNX format.
    export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)