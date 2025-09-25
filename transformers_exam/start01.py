from transformers import AutoTokenizer, AutoModelForCausalLM

# 使用一个免费可访问模型
model_name = "Qwen/Qwen2-0.5B-Instruct"

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print(f'{tokenizer=}')
print(f'{model=}')