import json
import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

_ = load_dotenv('../.env_examples')
api_key =os.getenv('OPENAI_API_KEY')
openai_url = os.getenv('OPENAI_BASE_URL')
print(f'{api_key=}\n{openai_url=}')

client = OpenAI(api_key=api_key,
                base_url=openai_url)
response = client.chat.completions.create(
    # model='Pro/deepseek-ai/DeepSeek-R1',
    # model="Qwen/Qwen2.5-72B-Instruct",
    # model="Qwen/QwQ-32B",
    model="deepseek-ai/DeepSeek-V2.5",
    messages=[
        {'role': 'user',
        'content': "推理模型会给市场带来哪些新的机会"},
        {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    ],
    # stream=True,
    response_format={'type': "json_object"}
)

print(f'')
print('-'*50)
print(type(response)) # <class 'openai.Stream'>

# for chunk in response:
#     # type(chunk)=<class 'openai.types.chat.chat_completion_chunk.ChatCompletionChunk'>
#     # response.response.__dict__={'status_code': 200, 'headers': Headers({'date': 'Wed, 01 Oct 2025 01:41:05 GMT', 'content-type': 'text/event-stream', 'transfer-encoding': 'chunked', 'connection': 'keep-alive', 'cache-control': 'no-cache'}), '_request': <Request('POST', 'https://api.siliconflow.cn/v1/chat/completions')>, 'next_request': None, 'extensions': {'http_version': b'HTTP/1.1', 'reason_phrase': b'OK', 'network_stream': <httpcore._backends.sync.SyncStream object at 0x7fc1c6c939a0>}, 'history': [], 'is_closed': False, 'is_stream_consumed': True, 'default_encoding': 'utf-8', 'stream': <httpx._client.BoundSyncStream object at 0x7fc1e881e490>, '_num_bytes_downloaded': 331, '_decoder': <httpx._decoders.IdentityDecoder object at 0x7fc1c6c93190>}
#     if not chunk.choices:
#         continue
#     # print(f'{len(chunk.choices)=}')
#     # print(f'{chunk.choices[0].delta.__repr__()=}')
#     # 推理(思考时):chunk.choices[0].delta.__repr__()="ChoiceDelta(content=None, function_call=None, refusal=None, role='assistant', tool_calls=None, reasoning_content='保持')"
#     # 生成时:推理chunk.choices[0].delta.__repr__()="ChoiceDelta(content='模型（', function_call=None, refusal=None, role='assistant', tool_calls=None, reasoning_content=None)"
#
#     if chunk.choices[0].delta.content:
#         # 同意对话每个id都一样
#         #  chunk.__repr__()="ChatCompletionChunk(id='01999d80f6a45bd5aeded44ead1922db', choices=[Choice(delta=ChoiceDelta(content=' - **', function_call=None, refusal=None, role='assistant', tool_calls=None, reasoning_content=None), finish_reason=None, index=0, logprobs=None)], created=1759284098, model='Qwen/QwQ-32B', object='chat.completion.chunk', service_tier=None, system_fingerprint='', usage=CompletionUsage(completion_tokens=700, prompt_tokens=18, total_tokens=718, completion_tokens_details=CompletionTokensDetails(accepted_prediction_tokens=None, audio_tokens=None, reasoning_tokens=620, rejected_prediction_tokens=None), prompt_tokens_details=None))"
#         # chunk.model='Qwen/QwQ-32B'
#         # 推理chunk.choices[0].__repr__()="Choice(delta=ChoiceDelta(content='模型（', function_call=None, refusal=None, role='assistant', tool_calls=None, reasoning_content=None), finish_reason=None, index=0, logprobs=None)"
#         # len(chunk.choices[0]) = 1
#         # print(f'{chunk.choices[0].delta.finish_reason=}')
#         # print(f'{type(chunk.choices[0])=}')
#         # type(chunk.choices[0])=<class 'openai.types.chat.chat_completion_chunk.Choice'>
#         print(chunk.choices[0].delta.content, end="", flush=True)
#     # 展示推理过程
#     if chunk.choices[0].delta.reasoning_content:
#         print(chunk.choices[0].delta.reasoning_content, end="", flush=True)


print(response.json())
# {
#   "id": "01999d8fe63f7ca89471a64b06b77dd5",
#   "choices": [
#     {
#       "finish_reason": "stop",
#       "index": 0,
#       "logprobs": null,
#       "message": {
#         "content": "{\n  \"response\": {\n    \"market_opportunities\": [\n      {\n        \"sector\": \"Healthcare\",\n        \"opportunities\": [\n          \"Personalized Medicine\",\n          \"Early Disease Detection\",\n          \"Drug Discovery Acceleration\"\n        ]\n      }\n,\n{\n  \"sector\": \"Finance\",\n  \"opportunities\": [\n    \"Algorithmic Trading\",\n    \"Risk Assessment\",\n    \"Customer Behavior Prediction\"\n  ]\n}\n,\n{\n  \"sector\": \"Retail\",\n  \"opportunities\": [\n    \"Demand Forecasting\",\n    \"Inventory Management\",\n    \"Customer Experience Enhancement\"\n  ]\n}\n,\n{\n  \"sector\": \"Manufacturing\",\n  \"opportunities\": [\n    \"Predictive Maintenance\",\n    \"Quality Control\",\n    \"Supply Chain Optimization\"\n  ]\n}\n,\n{\n  \"sector\": \"Education\",\n  \"opportunities\": [\n    \"Personalized Learning\",\n    \"Student Performance Prediction\",\n    \"Educational Content Recommendation\"\n  ]\n}\n,\n{\n  \"sector\": \"Customer Service\",\n  \"opportunities\": [\n    \"Chatbots\",\n    \"Sentiment Analysis\",\n    \"Issue Prediction and Resolution\"\n  ]\n}\n,\n{\n  \"sector\": \"Transportation\",\n  \"opportunities\": [\n    \"Traffic Flow Optimization\",\n    \"Autonomous Vehicles\",\n    \"Demand Prediction for Ride-Sharing\"\n  ]\n}\n,\n{\n  \"sector\": \"Entertainment\",\n  \"opportunities\": [\n    \"Content Personalization\",\n    \"Real-Time Content Creation\",\n    \"Audience Engagement Analysis\"\n  ]\n}\n    ]\n  }\n}",
#         "refusal": null,
#         "role": "assistant",
#         "annotations": null,
#         "audio": null,
#         "function_call": null,
#         "tool_calls": null
#       }
#     }
#   ],
#   "created": 1759285077,
#   "model": "deepseek-ai/DeepSeek-V2.5",
#   "object": "chat.completion",
#   "service_tier": null,
#   "system_fingerprint": "",
#   "usage": {
#     "completion_tokens": 396,
#     "prompt_tokens": 22,
#     "total_tokens": 418,
#     "completion_tokens_details": null,
#     "prompt_tokens_details": null
#   }
# }