import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from llm_from_scratch_cn.chapter2.mha_Mutli_version.CausalAttention_MHA_qkv_combined import MHA_qkv_Combined
from llm_from_scratch_cn.chapter2.mha_Mutli_version.CausalAttention_MHA import Ch03_MHA_Wrapper
from llm_from_scratch_cn.chapter2.mha_Mutli_version.CausalAttention_MHA_split_qkv import MHA
from llm_from_scratch_cn.chapter2.mha_Mutli_version.CausalAttention_einsum import MHAEinsum
from llm_from_scratch_cn.chapter2.mha_Mutli_version.scaled_dot_product_Attention_in_pytorch import Scaled_dot_Attention
from llm_from_scratch_cn.chapter2.mha_Mutli_version.scaled_dot_product_Attention_without_FlashAttn_in_pytorch import Scaled_dot_Attention_Without_FlashAttn


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

torch.manual_seed(123)
batch_size = 8
context_len = 1024
embed_dim = 768
embeddings = torch.randn((batch_size, context_len, embed_dim), device=device)
print(f'{embeddings.shape=}')
# [8, 1024, 768]
heads = 12

##########################

mha_split = MHA(d_in=embed_dim, d_out=embed_dim,context_length=context_len,dropout=0.0,num_heads=heads,qkv_bias=False).to(device)
mha_ch03_wrapper = Ch03_MHA_Wrapper(d_in=embed_dim,d_out=embed_dim // heads,context_length=context_len,dropout=0.0,num_heads=heads,qkv_bias=False).to(device)
mha_qkv_Combined = MHA_qkv_Combined(d_in=embed_dim, d_out=embed_dim,context_length=context_len,dropout=0.0,num_heads=heads,qkv_bias=False).to(device)
mha_einsum = MHAEinsum(d_in=embed_dim, d_out=embed_dim,context_length=context_len,dropout=0.0,num_heads=heads,qkv_bias=False).to(device)
scaled_dot_Attention = Scaled_dot_Attention(d_in=embed_dim, d_out=embed_dim,context_length=context_len,dropout=0.0,num_heads=heads,qkv_bias=False).to(device)
scaled_dot_Attention_Without_FlashAttn = Scaled_dot_Attention_Without_FlashAttn(d_in=embed_dim, d_out=embed_dim,context_length=context_len,dropout=0.0,num_heads=heads,qkv_bias=False).to(device)


functions = {
    "1) MHA wrapper class": mha_ch03_wrapper,
    "2) MHA Ch03": mha_split,
    "3) MHA with combined QKV weights": mha_qkv_Combined,
    "4) MHA with Einsum": mha_einsum,
    "5) MHA with PyTorch scaled_dot_product_attention": scaled_dot_Attention,
    "6) PyTorch's SDPA, no FlashAttention": scaled_dot_Attention_Without_FlashAttn,
    # "7) PyTorch MHA class defaults": mha_pytorch_class_default,
    # "8) PyTorch MHA with need_weights=False": mha_pytorch_class_noweights
    }



# Customize further for dark mode aesthetics
plt.rcParams["figure.facecolor"] = "#121212"
plt.rcParams["axes.facecolor"] = "#121212"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["grid.color"] = "#444444"
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["lines.markersize"] = 8

def plot_execution_times(functions, execution_means, execution_stds, filename, dpi: int = 300):

    # Create plot
    fig, ax = plt.subplots()
    # 绘制柱状图，yerr参数指定误差线（标准差），capsize设置误差线顶部和底部的横线长, error_kw是误差线的样式，设为‘灰色’
    bars = ax.bar(functions.keys(), execution_means, yerr=execution_stds, capsize=5, error_kw={'ecolor': 'grey'})

    plt.ylabel("Execution time (ms)")
    plt.xticks(rotation=45, ha="right") # 旋转45°

    # Calculate new ylim with a margin
    max_execution_time = max(execution_means)
    upper_ylim = max_execution_time + 0.4 * max_execution_time  # 增加40%的余量
    plt.ylim(0, upper_ylim)

    # Annotate bars with execution times
    # 在每个柱子上方标注执行时间
    for bar in bars:
        yval = bar.get_height() # 柱高
        # round(yval, 2): 标注的内容（保留两位小数的平均执行时间）
        plt.text(bar.get_x() + bar.get_width()/2, yval + (0.05 * upper_ylim), round(yval, 2), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(filename, dpi=dpi)
    plt.show()

def time_pytorch_function(func, *input, num_repeats=10):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Warmup
    for _ in range(5):
        func(*input)
    torch.cuda.synchronize()

    times = []
    for _ in tqdm(range(num_repeats)):
        start.record()
        func(*input)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return np.mean(times), np.std(times)

# num_repeats可自定
execution_stats = [time_pytorch_function(fn, embeddings, num_repeats=100) for fn in functions.values()]
execution_means = [stat[0] for stat in execution_stats] # 均值
execution_stds = [stat[1] for stat in execution_stats] # 方差


plot_execution_times(functions, execution_means, execution_stds, filename="1_forward-only.jpg", dpi=600)