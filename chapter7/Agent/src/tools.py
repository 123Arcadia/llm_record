import datetime

import requests
import wikipedia


def get_current_datetime() -> str:
    """
    获取当前日期时间
    :return:
    """
    cur_dt = datetime.datetime.now()
    format_dt = cur_dt.strftime("%Y-%m-%d H%:%M:%S")
    return format_dt

def add(a: float, b: float):
    """
    a 和 b 的相加
    :param a:
    :param b:
    :return:
    """
    return str(a+b)

def mul(a: float, b: float):
    """
    a和b的相乘
    :param a:
    :param b:
    :return:
    """
    return str(a*b)

def compare(a: float, b: float):
    """
    比较a和b的大小
    :param a:
    :param b:
    :return:
    """
    if a > b:
        return f'{a} is greater than {b}'
    elif a < b:
        return f'{b} is greater than {a}'
    else:
        return f'{a} is equal to {b}'

def count_letter_in_string(a, b: str):
    """
    统计字符串中某个字母出现的次数
    :param a: 字符串
    :param b: 搜索的字母
    :return: 返回出现的次数
    """
    string =  a.lower()
    letter =  b.lower()
    count = string.count(letter)
    return  (f"The letter '{letter}' appears {count} times in the string.")


def search_wikipedia(query: str) -> str:
    """
    在维基百科中搜索指定查询的前三个页面摘要。
    :param query: 要搜索的查询字符串。
    :return: 包含前三个页面摘要的字符串。
    """
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:  # 取前三个页面标题
        try:
            # 使用 wikipedia 模块的 page 函数，获取指定标题的维基百科页面对象。
            wiki_page = wikipedia.page(title=page_title, auto_suggest=False)
            # 获取页面摘要
            summaries.append(f"页面: {page_title}\n摘要: {wiki_page.summary}")
        except (
                wikipedia.exceptions.PageError,
                wikipedia.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "维基百科没有搜索到合适的结果"
    return "\n\n".join(summaries)


def get_current_temperature(latitude: float, longitude: float) -> str:
    """
    获取指定经纬度位置的当前温度。
    :param latitude: 纬度坐标。
    :param longitude: 经度坐标。
    :return: 当前温度的字符串表示。
    """
    open_meteo_url = "https://api.open-meteo.com/v1/forecast"
    # 请求参数
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1
    }

    response = requests.get(url=open_meteo_url, params=params)

    if response.status_code == 200:
        res = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    # 获取当前 UTC 时间
    current_utc_time = datetime.datetime.now(datetime.UTC)

    # 将时间字符串转换为 datetime 对象
    time_list = [datetime.datetime.fromisoformat(time_str).replace(tzinfo=datetime.timezone.utc)
                 for time_str in res['hourly']['time']]

    # 获取温度列表
    temperature_list = res['hourly']['temperature_2m']

    # 找到最接近当前时间的索引
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))

    # 获取当前温度
    current_temperature = temperature_list[closest_time_index]

    # 返回当前温度的字符串形式
    return f'现在温度是 {current_temperature}°C'











