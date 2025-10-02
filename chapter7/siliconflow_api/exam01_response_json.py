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
    response_format={"type": "json_object"}
)

print(type(response))
print(response.json())
# {
#   "id": "01999d8b897bd9defb101bd4ccfa10a2",
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "logprobs": null,
#       "message": {
#         "content": "{\"男子冠军\": \"马龙\", \"女子冠军\": \"陈梦\"}",
#         "refusal": null,
#         "role": "assistant",
#         "annotations": null,
#         "audio": null,
#         "function_call": null,
#         "tool_calls": null
#       }
#     }
#   ],
#   "created": 1759284791,
#   "model": "deepseek-ai/DeepSeek-V2.5",
#   "object": "chat.completion",
#   "service_tier": null,
#   "system_fingerprint": "",
#   "usage": {
#     "completion_tokens": 16,
#     "prompt_tokens": 50,
#     "total_tokens": 66,
#     "completion_tokens_details": null,
#     "prompt_tokens_details": null
#   }
# }