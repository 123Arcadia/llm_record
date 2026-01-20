import os

import torch



def test_DataLoader():
    ds = torch.arange(12)
    data_loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    for i, item in enumerate(data_loader):
        data_loader.desc = f'{i=} ...'
        print(f'{i=} {item=}  ')
        print(f'{data_loader.desc}')

        
    print()
    print(f'{data_loader.desc}')
    print(f'{__name__}')

    print(os.getenv("CUDA_VISIBLE_DEVICES"))

    rank = os.getenv("LOCAL_RANK")
    acc = torch.accelerator.current_accelerator()
    # configure map_location properly
    map_location = {f'{acc}:0': f'{acc}:{rank}'}
    print(f'{acc=}')
    print(f'{map_location=}')
    # acc=device(type='cuda')
    # map_location={'cuda:0': 'cuda:None'}

