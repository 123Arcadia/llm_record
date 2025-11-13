# !wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1a0sf5C209CLW5824TJkUM4olMy0zZWpg' -O fake_sft.json
from json import JSONDecodeError


def read_first_n_json_lines(file_path, n=5):
    """
    读取JSON Lines格式文件的前n个数据
    JSON Lines格式：每行一个独立的JSON对象
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if count >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                json_obj = json.loads(line)
                data.append(json_obj)
                count += 1
            except JSONDecodeError:
                print(f"跳过无效的JSON行: {line}")
    return data


if __name__ == '__main__':

    path = './fake_sft.json'
    json = read_first_n_json_lines(path, 5)
    print(f'{json=}')