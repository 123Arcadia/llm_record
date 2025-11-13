import torch


def test_eq():
    a = torch.arange(0 ,4).reshape(1, 4)
    b = torch.arange(0 ,4).reshape(1, 4)
    b[0][1] +=1
    print(f'{a=}')
    print(f'{b=}')
    print(f'{a.eq(b).sum()}')
    # a=tensor([[0, 1, 2, 3]])
    # b=tensor([[0, 2, 2, 3]])
    # 3

def test_act_fun():
    import  torch.nn.functional as F
    input = torch.arange(0, 4, dtype=torch.float16).reshape(1, 4)
    target = torch.arange(0, 4, dtype=torch.float16).reshape(1, 4)
    input = F.log_softmax(input)
    target = F.log_softmax(target)
    target[0][2] += 1
    print(f'{input=}')
    print(f'{target=}')
    f1 = F.mse_loss(input, target)
    f2 = F.binary_cross_entropy(input, target)
    print(f1.eq(f2).sum())
    print(f'{f1=}')
    print(f'{f2=}')

def test_topk():
    # a = torch.arange(0, 9).reshape(3,3).float()
    a = torch.arange(0, 9).reshape(1, -1).float()
    out_v, out_i = torch.topk(a, 2, dim=-1)
    print()
    print(f'{out_v=} {out_v.shape=}')
    print(f'{out_i=} {out_i.shape=}')
    # dim=-1
    # out=torch.return_types.topk(
    # values=tensor([[2., 1.],
    #         [5., 4.],
    #         [8., 7.]]),
    # indices=tensor([[2, 1],
    #         [2, 1],
    #         [2, 1]]))

    # dim=0
    # out=torch.return_types.topk(
    # values=tensor([[6., 7., 8.],
    #         [3., 4., 5.]]),
    # indices=tensor([[2, 2, 2],
    #         [1, 1, 1]]))
    print(out_v[:, [-1]])
    # dim=-1
    # tensor([[1.],
    #         [4.],
    #         [7.]])
    print(out_v[:, -1])
    # tensor([1., 4., 7.])



def test_nllLoss_logSoftmax():
    import torch
    import torch.nn.functional as F
    print()
    # 模型输出的原始 logits（批量大小=2，类别数=3）
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # 步骤1：对 logits 应用 log_softmax，得到对数概率
    log_probs = F.log_softmax(logits, dim=1)
    # log_probs 输出（每行和为 log(1) = 0，符合概率归一化）：
    # tensor([[-2.4076, -1.4076, -0.4076],
    #         [-2.4076, -1.4076, -0.4076]])

    # 真实标签（类别索引）
    target = torch.tensor([2, 2])  # 两个样本的真实类别都是索引 2

    # 步骤2：用 nll_loss 计算损失
    loss = F.nll_loss(log_probs, target)
    print(loss)  # 输出：tensor(0.4076)


    # 使用交叉熵
    loss2 = F.cross_entropy(logits, target)
    print(f"{loss == loss2}")
    # True