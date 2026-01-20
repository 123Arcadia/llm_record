import torch
import torch.nn as nn
from convert_gpt_to_llama2 import RMSNorm, MultiHeadAttention
from llm_from_scratch_cn.chapter5.gpt_to_llama_07.convert_gpt_to_llama2 import precompute_rope_params, compute_rope


def test_RMSNorm():
    torch.manual_seed(123)

    example_batch = torch.randn(2, 3, 4)

    rms_norm = RMSNorm(emb_dim=example_batch.shape[-1])
    rmsnorm_pytorch = nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

    assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch)), "torch.allclose通过失败"
    assert torch.equal(rms_norm(example_batch), rmsnorm_pytorch(example_batch)), "torch.aequal通过失败"


def test_rope():
    bsz = 2
    context_length = 5
    num_heads = 4
    head_dim = 16
    cos, sin = precompute_rope_params(head_dim, context_length=context_length)
    # precompute_freqs_cis()
    torch.manual_seed(123)
    q = torch.randn(bsz, num_heads, context_length, head_dim)  # (2, 4, 5, 16)
    k = torch.randn(bsz, num_heads, context_length, head_dim)

    q_dot = compute_rope(q, cos, sin)
    k_dot = compute_rope(k, cos, sin)
    print(f'{q_dot.shape=}')
    # q_dot.shape=torch.Size([2, 4, 5, 16])
    print(f'{k_dot.shape=}')
    # k_dot.shape=torch.Size([2, 4, 5, 16])


def test_rope01():
    """
    # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
    """
    print()
    a = torch.arange(10).unsqueeze(0)
    print(f'{a.shape=}')  # [1, 10]
    b = torch.stack([-a[..., 1::2], a[..., ::2]], dim=-1).reshape_as(a)
    print(f'{b}')
    print(f'{b.shape=}')
    # tensor([[-1,  0, -3,  2, -5,  4, -7,  6, -9,  8]])
    # b.shape=torch.Size([1, 10])
    c = torch.stack([-a[..., 1::2], a[..., ::2]], dim=-1)
    print(f'{c=}')
    # c=tensor([[[-1,  0],
    #          [-3,  2],
    #          [-5,  4],
    #          [-7,  6],
    #          [-9,  8]]]) # (5, 2)
    print(f'{c.reshape(1, 10)}')
    # tensor([[-1,  0, -3,  2, -5,  4, -7,  6, -9,  8]])


def test_triu():
    context_length = 5
    a = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    print()
    print(a)
    # tensor([[0., 1., 1., 1., 1.],
    #         [0., 0., 1., 1., 1.],
    #         [0., 0., 0., 1., 1.],
    #         [0., 0., 0., 0., 1.],
    #         [0., 0., 0., 0., 0.]])

def test_mha_llama():
    # Settings
    batch_size = 1
    context_len = 100
    max_context_len = 4096
    embed_dim = 128
    num_heads = 4
    torch.manual_seed(123)
    example_batch = torch.randn((batch_size, context_len, embed_dim))

    mha = MultiHeadAttention(
        d_in=embed_dim,
        d_out=embed_dim,
        context_length=max_context_len,
        num_heads=num_heads
    )
    out = mha(example_batch)
    print(f'{out.shape=}') # out.shape=torch.Size([1, 100, 128])

def test_repeat_interleave():
    a = torch.arange(0, 5).reshape(1, 5)
    print()
    print(f'{a.shape=}')
    a1 = a.repeat(2, 1) # dim=0重复2次，dim=1重复1次
    print(a1)
    print(f'{a1.shape=}')
    a2 = a.repeat_interleave(2, 1)  # dim=0重复2次，dim=1重复1次
    print(a2)
    print(f'{a2.shape=}')
    # a.shape=torch.Size([1, 5])
    # tensor([[0, 1, 2, 3, 4],
    #         [0, 1, 2, 3, 4]])
    # a1.shape=torch.Size([2, 5])
    # tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]])
    # a2.shape=torch.Size([1, 10])

def test_tensor_contiguous():
    torch.manual_seed(123)
    a = torch.randn(2, 3, 4)
    # print(f'{a[:, -1, :].shape=}') # torch.Size([2, 4])
    print(f'{a.shape=}')
    a1  = a.transpose(1 ,2)
    print(f'{a1=}')
    print(f'{a1.shape=}')
    print(f'{a1.is_contiguous()=}')

    a2 = a.view(2, 4, 3)
    print(f'{a2=}')
    print(f'{a2.shape=}')
    print(f'{a2.is_contiguous()=}')

    a3 = a.reshape(2, 4, 3)
    print(f'{a3=}')
    print(f'{a3.shape=}')
    print(f'{a3.is_contiguous()=}')



















