import requests

url = "https://api.siliconflow.cn/v1/chat/completions"

payload = {
    "thinking_budget": 4096,
    "top_p": 0.7
}
headers = {
    "Authorization": "Bearer <token>",
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)
print(f'回答:')
print(response.json())