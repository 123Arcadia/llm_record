
import os

from dotenv import load_dotenv
from openai import OpenAI

_ = load_dotenv('../.env_examples')
api_key = os.getenv('OPENAI_API_KEY')
openai_url = os.getenv('OPENAI_BASE_URL')
print(f'{api_key=}\n{openai_url=}')
client = OpenAI(api_key=api_key, base_url=openai_url)


response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V2.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
        {"role": "user", "content": "? 2020 年世界奥运会乒乓球男子和女子单打冠军分别是谁? "
                                    "Please respond in the format {\"男子冠军\": ..., \"女子冠军\": ...}"}
    ],
    tools=[
        {
            'type': 'function',
            'function': {
                'name': '对应到实际执行的函数名称',
                'description': '此处是函数相关描述',
                'parameters': {
                   # 此处是函数参数相关描述
                },
            }
        },
        {
            # 其他函数相关说明
        }
    ],
    response_format={'type':'json_object'}
    #  chat.completions 其他参数
)

print(response)