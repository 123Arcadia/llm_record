import os


def test_os_listdir():
    ctx = os.listdir('./autodl-tmp')
    print(f'{ctx=}')
    # ctx=['dataset', 'output', 'qwen-1.5b']
    if os.path.isdir(os.path.join('./autodl-tmp', ctx[0])):
        print(f'ctx[0]是dir')
        # ctx[0]是dir