

def test_color_print():
    print(f'\033[94m zzUser') # 蓝色
    # zzUser

def test_to_dict():
    str = '{"a": 9.01, "b": 9.1}'
    print(eval(str))

def test_chr_func():
    str = chr(10).join("im a student")
    print(str)

def test_append():
    a = ['1', '2']
    b = ['11', '22']
    print(a + b)
    # ['1', '2', '11', '22']