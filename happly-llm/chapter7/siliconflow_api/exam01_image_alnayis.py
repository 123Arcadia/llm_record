import os

from dotenv import load_dotenv
from openai import OpenAI


_ = load_dotenv('../.env_examples')
api_key =os.getenv('OPENAI_API_KEY')
openai_url = os.getenv('OPENAI_BASE_URL')
print(f'{api_key=}\n{openai_url=}')
client = OpenAI(api_key=api_key, base_url=openai_url)

response = client.chat.completions.create(
    model="deepseek-ai/deepseek-vl2",
    messages=[
        {
            "role": "user",
             "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://sf-maas-uat-prod.oss-cn-shanghai.aliyuncs.com/outputs/658c7434-ec12-49cc-90e6-fe22ccccaf62_00001_.png",
                        },
                    },
                     {
                         "type": "text",
                         # "text": "What's in this image?"
                         "text": "这张图片是什么？"
                     }
                ],
        }
    ],
    temperature=0.7,
    max_tokens=1024,
    stream=True
)
# 逐步接收并处理响应
for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)