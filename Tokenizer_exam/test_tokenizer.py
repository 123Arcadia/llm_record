import json
import os

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
    print(f'{model.parameters()=}')
    for p in model.parameters():
        print(f'{p} {type(p)}') # 本质就是tensor
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


def test_input_ids():
    # input_ids 形状：[batch_size, seq_len]（如 [2, 10]，2个样本，长度10）
    input_ids = torch.tensor([[1, 5, 3, 2, 0, 0], [1, 8, 6, 4, 2, 0]])  # 1: bos, 2: eos, 0: pad

    # 输入序列：取 input_ids 除最后一个 token 外的所有部分（用于预测）
    inputs = input_ids[:, :-1]  # 形状 [2, 5]

    # 目标序列：取 input_ids 除第一个 token 外的所有部分（需要预测的真实值）
    labels = input_ids[:, 1:]  # 形状 [2, 5]
    print(f'{inputs} {inputs.shape}')
    print(f'{labels} {labels.shape}')
    # tensor([[1, 5, 3, 2, 0],
    #         [1, 8, 6, 4, 2]]) torch.Size([2, 5])
    # tensor([[5, 3, 2, 0, 0],
    #         [8, 6, 4, 2, 0]]) torch.Size([2, 5])

def test_get_env_info():
    for k, v in os.environ.__dict__.items():
        print(f'{k}  {v}')
    #     _data  {b'SHELL': b'/bin/bash', b'PYTHONUNBUFFERED': b'1', b'JAVA_PATH': b'/usr/local/java/jdk-17/bin:/usr/local/java/jdk-17/jre/bin', b'WSL2_GUI_APPS_ENABLED': b'1', b'CONDA_MKL_INTERFACE_LAYER_BACKUP': b'', b'CONDA_EXE': b'/home/zhangchenwei/miniconda3/bin/conda', b'_CE_M': b'', b'WSL_DISTRO_NAME': b'Ubuntu-22.04', b'PYCHARM_DISPLAY_PORT': b'52165', b'JAVA_HOME': b'/usr/local/java/jdk-17', b'JRE_HOME': b'/usr/local/java/jdk-17/jre', b'XML_CATALOG_FILES': b'file:///home/zhangchenwei/miniconda3/envs/torch2.4.1/etc/xml/catalog file:///etc/xml/catalog', b'NAME': b'LAPTOP-89JUTCMV', b'PWD': b'/home/zhangchenwei/llm_record/Tokenizer_exam', b'LOGNAME': b'zhangchenwei', b'CONDA_ROOT': b'/home/zhangchenwei/miniconda3', b'CONDA_PREFIX': b'/home/zhangchenwei/miniconda3/envs/torch2.4.1', b'HOME': b'/home/zhangchenwei', b'LANG': b'zh_CN.UTF-8', b'WSL_INTEROP': b'/run/WSL/46_interop', b'WAYLAND_DISPLAY': b'wayland-0', b'CONDA_PROMPT_MODIFIER': b'(torch2.4.1) ', b'PYCHARM_HELPERS_DIR': b'/home/zhangchenwei/.pycharm_helpers/pycharm', b'PYTEST_RUN_CONFIG': b'True', b'PYTHONPATH': b'/home/zhangchenwei/.pycharm_helpers/pycharm:/home/zhangchenwei/llm_record:/home/zhangchenwei/.pycharm_helpers/pycharm_plotly_backend:/home/zhangchenwei/.pycharm_helpers/pycharm_matplotlib_backend:/home/zhangchenwei/.pycharm_helpers/pycharm_display', b'TERM': b'xterm-256color', b'_CE_CONDA': b'', b'USER': b'zhangchenwei', b'PYTHONIOENCODING': b'UTF-8', b'CONDA_SHLVL': b'2', b'DISPLAY': b':0', b'SHLVL': b'1', b'CONDA_PYTHON_EXE': b'/home/zhangchenwei/miniconda3/bin/python', b'CLASSPATH': b'.:/usr/local/java/jdk-17/lib:/usr/local/java/jdk-17/jre/lib:', b'PYCHARM_PROJECT_ID': b'472c1dea', b'XDG_RUNTIME_DIR': b'/mnt/wslg/runtime-dir', b'CONDA_DEFAULT_ENV': b'torch2.4.1', b'PYCHARM_INTERACTIVE_PLOTS': b'1', b'WSLENV': b'', b'XDG_DATA_DIRS': b'/usr/local/share:/usr/share:/var/lib/snapd/desktop', b'PATH': b'/home/zhangchenwei/miniconda3/envs/torch2.4.1/bin:/home/zhangchenwei/miniconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/lib/wsl/lib:/mnt/e/java/jdk_8u381/lib:/mnt/e/java/jdk_8u381/bin:/mnt/d/CSProject/apache-maven-3.8.6-bin/apache-maven-3.8.6/bin:/mnt/c/Windows/system32:/mnt/c/Windows:/mnt/c/Windows/System32/Wbem:/mnt/c/Windows/System32/WindowsPowerShell/v1.0:/mnt/c/Windows/System32/OpenSSH:/mnt/c/Program Files (x86)/NVIDIA Corporation/PhysX/Common:/mnt/c/Program Files (x86)/Common Files/Oracle/Java/javapath:/mnt/c/WINDOWS/system32:/mnt/c/WINDOWS:/mnt/c/WINDOWS/System32/Wbem:/mnt/c/WINDOWS/System32/WindowsPowerShell/v1.0:/mnt/c/WINDOWS/System32/OpenSSH:/mnt/c/x86_64-8.1.0-release-win32-seh-rt_v6-rev0/mingw64/bin:/mnt/d/Git/Git/cmd:/mnt/c/Cmake/bin:/mnt/d/MysqlServer8.0.31.0/bin:/mnt/d/\xe8\xb0\xb7\xe6\xad\x8c\xe4\xb8\x8b\xe8\xbd\xbd/pdftk_server/PDFtk/bin:/mnt/c/Program Files/Microsoft SQL Server/Client SDK/ODBC/170/Tools/Binn:/mnt/c/Program Files (x86)/Microsoft SQL Server/150/Tools/Binn:/mnt/c/Program Files/Microsoft SQL Server/150/Tools/Binn:/mnt/c/Program Files/Microsoft SQL Server/150/DTS/Binn:/mnt/c/Program Files/Azure Data Studio/bin:/mnt/c/Program Files (x86)/Microsoft SQL Server/150/DTS/Binn:/mnt/c/Program Files/dotnet:/mnt/c/ProgramData/chocolatey/bin:/mnt/d/NodeJs_20.11.0:/mnt/d/NodeJs_20.11.0/node_global/node_modules:/mnt/d/Anaconda3:/mnt/d/Anaconda3/Library/bin:/mnt/d/Anaconda3/Scripts:/mnt/e/VsCode/Microsoft VS Code/bin:/mnt/e/Windows Kits/10/Windows Performance Toolkit:/mnt/c/Program Files/Docker/Docker/resources/bin:/mnt/c/x86_64-8.1.0-release-win32-seh-rt_v6-rev0/mingw64/bin:/mnt/c/Users/zhangchenwei/AppData/Local/Microsoft/WindowsApps:/mnt/d/PyCharm/PyCharm 2023.1/bin:/mnt/e/java/jdk_8u381/bin:/mnt/d/GoLand/GoSDK/bin:/mnt/f/ffmeg/ffmpeg-2023-09-07-git-9c9f48e7f2-essentials_build/ffmpeg-2023-09-07-git-9c9f48e7f2-essentials_build/bin:/mnt/d/NodeJs_20.11.0/node_global:/mnt/c/Users/zhangchenwei/.poetry/bin:/mnt/c/Program Files/Azure Data Studio/bin:/mnt/d/Anaconda3:/mnt/d/Anaconda3/Scripts:/mnt/d/Anaconda3/Library/bin:/mnt/d/Clion/CLion 2024.2/bin:/mnt/c/Cmake/bin:/mnt/f/texlive/2024/bin/windows:/snap/bin:/usr/local/java/jdk-17/bin:/usr/local/java/jdk-17/jre/bin', b'HOSTTYPE': b'x86_64', b'CONDA_PREFIX_1': b'/home/zhangchenwei/miniconda3', b'PYCHARM_HOSTED': b'1', b'PULSE_SERVER': b'unix:/mnt/wslg/PulseServer', b'OLDPWD': b'/tmp/tmp.7AoUrnuI0t', b'MKL_INTERFACE_LAYER': b'LP64,GNU', b'_': b'/home/zhangchenwei/miniconda3/envs/torch2.4.1/bin/python', b'TEAMCITY_VERSION': b'LOCAL', b'_JB_PPRINT_PRIMITIVES': b'1', b'PYTEST_VERSION': b'8.3.5', b'PYTEST_CURRENT_TEST': b'test_tokenizer.py::test_get_env_info (call)'}
    #


def test_repeat_tensor():
    a  = torch.arange(0,9).reshape(1,3,3)
    print(f'{a.shape=}')
    a1 = a.expand(2,3,3)
    print(f'{a1.shape=}')


def test_unsqueeze():
    a=torch.randn(1,2,2)
    a1 = a.unsqueeze(2)
    print(f'{a1.shape}')
    # torch.Size([1, 2, 1, 2])
    a2 = a.unsqueeze(3)
    print(f'{a2.shape}')
    # torch.Size([1, 2, 2, 1])

def test_insert_None():
    a=torch.randn(1,2,2)
    print(f'{a.shape}')
    a2 = a[:,:,None,:]
    print(f'{a2.shape}')
#     torch.Size([1, 2, 1, 2])
