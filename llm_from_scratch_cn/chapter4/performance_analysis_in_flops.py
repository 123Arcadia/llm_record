
from importlib.metadata import version

import torch
from thop import profile

from llm_from_scratch_cn.sft_gpt2.additional_experiments.pervious_chapters import GPTModel

pkgs = [
    "thop",
    "torch",
]
for p in pkgs:
    print(f"{p} version: {version(p)}")



BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
input_tensor = torch.randint(0, 50257, (batch_size, 1024)).to(device)
# print(f'{input_tensor.shape=}')
# input_tensor.shape=torch.Size([2, 1024])
for size in model_configs:
    BASE_CONFIG.update(model_configs[size])

    model = GPTModel(BASE_CONFIG).bfloat16()
    model.to(device)
    total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    out_head_params = sum([p.numel() for p in model.out_head.parameters() if p.requires_grad])
    re_total_params = total_params - out_head_params
    print(f'total_params: {total_params/1024/1024:.2f} MB, out_head_params: {out_head_params/1024/1024:.2f} MB'
          f', re_total_params: {re_total_params/1024/1024:.2f} MB')

    # MACS = multiply-accumulate operations
    # MACS are typically counted as two FLOPS (one multiply and one accumulate)
    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2*macs
    print(f"{size:18}: {flops:.1e} FLOPS, params: {params/1024/1024:.2f}")

    # 如果需要GFLOPs
    # print("FLOPs=", str(flops / 1e9) + "G")  # 打印FLOPs，单位为G
    #
    # print("params=", str(params / 1e6) + "M")  # 打印参数量，单位为M

    del model
    torch.cuda.empty_cache()



# gpt-small (124M)  : 5.1e+11 FLOPS, params: 117.89 MB
# gpt-medium (355M) : 1.4e+12 FLOPS, params: 337.29 MB
# gpt-large (774M)  : 3.2e+12 FLOPS, params: 736.74 MB
# gpt-xl (1558M)    : 6.4e+12 FLOPS, params: 1483.60 MB

# 附加p.numel()自己计算的
# total_params: 155.48 MB, out_head_params: 36.81 MB, re_total_params: 118.68 MB
# gpt-small (124M)  : 5.1e+11 FLOPS, params: 117.89

# total_params: 387.46 MB, out_head_params: 49.08 MB, re_total_params: 338.39 MB
# gpt-medium (355M) : 1.4e+12 FLOPS, params: 337.29

# total_params: 799.52 MB, out_head_params: 61.35 MB, re_total_params: 738.17 MB
# gpt-large (774M)  : 3.2e+12 FLOPS, params: 736.74

# total_params: 1562.14 MB, out_head_params: 76.69 MB, re_total_params: 1485.45 MB
# gpt-xl (1558M)    : 6.4e+12 FLOPS, params: 1483.60
# 可以看到减去out_head层的参数后大概一样










