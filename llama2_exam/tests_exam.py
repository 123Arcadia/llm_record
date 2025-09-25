import torch
from torch import nn

from llama_attn_exam import apply_rotary_emb, precompute_freqs_cis, ModelConfig, llama_Attn, MLP, DecoderLayer, \
    Transformer


def test_apply_rotary_emb():
    xq = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim
    xk = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim
    # 使用 precompute_freqs_cis 函数获取 sin和cos
    cos, sin = precompute_freqs_cis(288 // 6, 50)
    print(cos.shape, sin.shape)
    # torch.Size([50, 24]) torch.Size([50, 24])
    xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)

    print(f'{xq_out.shape} {xk_out.shape}')
    # torch.Size([1, 50, 6, 48]) torch.Size([1, 50, 6, 48])

def test_xy():
    xq = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim
    xk = torch.randn(1, 50, 6, 48)  # bs, seq_len, dim//n_head, n_head_dim
    sh = xq.shape[:-1] + (-1, 2)
    print(f'{sh=}')
    # sh=torch.Size([1, 50, 6, -1, 2])
    sh1 = xq.unbind(-1) #得到tuple
    print(f'{len(sh1)=}') # 48 哥 shape=[1,50,6]的tensor
    for i, s in enumerate(sh1):
        print(f'{i=} {s.shape=}')


def test_matcdot():
    x = torch.arange(0, 8).reshape(2,4)
    y = torch.arange(0, 4).reshape(1,4)
    o1 = x*y # 逐位相乘: y 会广播
    print(f'{o1.shape=}')
    # o1.shape=torch.Size([2, 4])


def test_llama_attn():
    args = ModelConfig()
    print(f'{args=}')
    attn_model = llama_Attn(args)
    bsz = 1
    seq_len = 50
    dim = args.dim
    x = torch.rand(bsz, seq_len, dim)
    # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
    # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

    freqs_cos, freqs_sin = precompute_freqs_cis(dim // args.n_heads, seq_len)

    # 运行Attention模型
    output = attn_model(x, freqs_cos, freqs_sin)

    # attention出来之后的形状 依然是[batch_size, seq_len, dim]
    print("Output shape:", output.shape)
    # xq_r.shape=torch.Size([1, 50, 16, 24])  xq_i.shape=torch.Size([1, 50, 16, 24])
    # (xq_r * freqs_cos).shape=torch.Size([1, 50, 16, 24])
    # xq_out_r.shape=torch.Size([1, 50, 16, 24])
    # xq_out.shape=torch.Size([1, 50, 16, 48])
    # Output shape: torch.Size([1, 50, 768])

def test_llama_mlp():
    args = ModelConfig()
    # 创建MLP实例
    print(f'params: {args.dim} {args.hidden_dim} {args.multiple_of} {args.dropout}')
    # params: 768 None 64 0.0
    mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
    # 随机生成数据
    x = torch.randn(1, 50, args.dim)
    # 运行MLP模型
    output = mlp(x)
    print(output.shape)
    # torch.Size([1, 50, 768])

def test_attn_feed():
    # 创建LLaMADecoderLayer实例
    args = ModelConfig()
    decoderlayer = DecoderLayer(0, args)

    # 模拟输入数据
    dim = args.dim
    seq_len = 50

    x = torch.randn(1, seq_len, dim)  # [bs, seq_len, dim]

    freqs_cos, freqs_sin = precompute_freqs_cis(dim // args.n_heads, seq_len)

    out = decoderlayer.forward(x, freqs_cos, freqs_sin)

    print(out.shape)  # 形状和输入的x一样 [batch_size, seq_len, dim]

def test_emb():
    xw = nn.Linear(10 ,24, bias=False)
    mod = [n_mod for i, n_mod in enumerate(xw.named_modules())]
    param = [n_param for i, n_param in enumerate(xw.named_parameters())]
    print(f'{mod=}')
    print(f'{param=}')
    # mod=[('', Linear(in_features=10, out_features=24, bias=False))]
    # param=[('weight', Parameter containing:
    # tensor


def test_llama_attn():
    # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
    x = torch.randint(0, 6144, (1, 50))  # [bs, seq_len]
    # 实例化LLaMA2Model
    args = ModelConfig()
    model = Transformer(args=args)
    # 计算model的全部参数
    num_params = sum(p.numel() for p in model.parameters())
    print('Number of parameters:', num_params)

    out = model(x)
    print(out.logits.shape)  # [batch_size, 1, vocab_size]





















