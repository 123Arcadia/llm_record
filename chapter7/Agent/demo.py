from dotenv import load_dotenv
from openai import OpenAI
import os


from chapter7.Agent.src.core import Agent
from chapter7.Agent.src.tools import get_current_datetime, search_wikipedia, get_current_temperature, add,add, compare, count_letter_in_string

_ = load_dotenv('../.env_examples')
api_key = os.getenv('OPENAI_API_KEY')
openai_url = os.getenv('OPENAI_BASE_URL')
print(f'{api_key=}\n{openai_url=}')

if __name__ == '__main__':
    assert api_key is not None and openai_url is not None, "api_key或url设置错误!"
    client = OpenAI(api_key=api_key, base_url=openai_url)
    agent = Agent(
        client=client,
        model='Qwen/Qwen2.5-32B-Instruct',
        tools = [get_current_datetime, add, compare, count_letter_in_string],
        verbose=True,
    )

    while True:
        prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
        if prompt.lower() == "exit":
            break
        response = agent.get_completion(prompt)
        print('\033[92mAssistant: \033[0m', response) # 绿色显示AI助手回答