

def test_color_print():
    print(f'\033[94m zzUser') # 蓝色
    # zzUser

def test_to_dict():
    str = '{"a": 9.01, "b": 9.1}'
    print(eval(str))