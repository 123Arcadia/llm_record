import json
from datetime import datetime

import torch
import uvicorn
from fastapi import FastAPI,Request

from transformers import AutoTokenizer, AutoModelForCausalLM

# device
DEVICE = "cuda"
DEVICE_ID = "0"
# 默认指定第一块GPU
CUDA_DEVICE = F'{DEVICE}:{DEVICE_ID}' if DEVICE_ID else DEVICE


# 清理GPU内存函数
def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()  # 清空CUDA缓存
            torch.cuda.ipc_collect()  # 收集CUDA内存碎片


app = FastAPI()


@app.post("/create_item")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()  # 获取POST请求的json数据
    json_post = json.dumps(json_post_raw)  # json转为字符串
    jsong_post_list = json.loads(json_post)  # 字符串转为py对象
    prompt = jsong_post_list.get('prompt')

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    # 调用模型
    input_ids = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors='pt').to(DEVICE)
    generated_ids = model.generate(model_inputs.input_ids, max_new_token=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    now = datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    # 构建响应json
    answer = {
        'response': response,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(f'{log=}')
    torch_gc()
    return answer




if __name__ == '__main__':
    model_name_or_path = '/root/autodl-tmp/qwen/Qwen1.5-7B-Chat'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)

    # 启动fastapi
    uvicorn.run(app, host='0.0.0.0', port=6006, workers=1)
