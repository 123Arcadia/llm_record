import json

import torch
from torch import nn, optim


def test_enumerate_file():
    with open('./tokenizer_k/special_tokens_map.json', 'r', encoding='utf-8') as f:
        # for line_num, line in enumerate(f, 0):
        #     # print(f'{line_num=}, {line=}')
        out = json.load(f) # 适用文件对象
        # out = json.loads("***") 适用字符串
        print(f'{out=}')


def test_last_dim():
    a = torch.arange(0, 9).reshape(1,3,3)
    print(f'{a[:, -1,:].shape}')
    # tensor([[6, 7, 8]])  shape=torch.Size([1, 3])


def test_optim_params_group():
    model = nn.Linear(4,4)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    print(f'{optimizer=}')
    print(f'{optimizer.param_groups=}')
    # optimizer=Adam (
    # Parameter Group 0
    #     amsgrad: False
    #     betas: (0.9, 0.999)
    #     capturable: False
    #     differentiable: False
    #     eps: 1e-08
    #     foreach: None
    #     fused: None
    #     lr: 0.001
    #     maximize: False
    #     weight_decay: 0
    # )
    # optimizer.param_groups=[{'params': [Parameter containing:
    # tensor([[ 0.1531,  0.0727, -0.2806, -0.3128],
    #         [ 0.3615,  0.0296, -0.1738,  0.0166],
    #         [ 0.2940, -0.2058, -0.4211,  0.4848],
    #         [-0.3463, -0.1916,  0.0728,  0.2502]], requires_grad=True), Parameter containing:
    # tensor([-0.2866, -0.0692,  0.2190,  0.0272], requires_grad=True)], 'lr': 0.001,
    # 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False,
    # 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}]
    print(f'{ optimizer.__dict__=}')
    # optimizer.__dict__={'defaults': {'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False, 'fused': None}, '_optimizer_step_pre_hooks': OrderedDict(), '_optimizer_step_post_hooks': OrderedDict(), '_optimizer_state_dict_pre_hooks': OrderedDict(), '_optimizer_state_dict_post_hooks': OrderedDict(), '_optimizer_load_state_dict_pre_hooks': OrderedDict(), '_optimizer_load_state_dict_post_hooks': OrderedDict(), '_zero_grad_profile_name': 'Optimizer.zero_grad#Adam.zero_grad', 'state': defaultdict(<class 'dict'>, {}), 'param_groups': [{'params': [Parameter containing:
    # tensor([[ 0.1861,  0.0490,  0.1096, -0.2162],
    #         [-0.3974,  0.1096, -0.0857,  0.2477],
    #         [-0.2874,  0.1965, -0.4180, -0.3067],
    #         [ 0.0105, -0.2985,  0.2274,  0.1554]], requires_grad=True), Parameter containing:
    # tensor([ 0.0095, -0.2240, -0.3767, -0.1637], requires_grad=True)], 'lr': 0.001,
    # 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0, 'amsgrad': False,
    # 'maximize': False, 'foreach': None, 'capturable': False, 'differentiable': False,
    # 'fused': None}], '_warned_capturable_if_run_uncaptured': True}


def test_io_readlines():
    path = './tokenizer_k/special_tokens_map.json'
    with open(path, 'r') as f:
        data = f.readlines()
        print(f'{data=}')