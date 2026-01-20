import argparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="local rank")
    parser.add_argument("--local_rank", type=int, default=-1)
    arg = parser.parse_args()

    torch.cuda.set_device(arg.local_rank)
    print("local rank: ", arg.local_rank)
    print(f"{torch.cuda.device}")
    # local rank:  0
    # <class 'torch.cuda.device'>
    print(torch.cuda.device_count())
