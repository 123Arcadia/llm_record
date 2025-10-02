import importlib
import inspect
from typing import List, Dict, Any

from openai import OpenAI

from chapter7.Agent.src.utils import function_to_json

# from chapter7.Agent.src.tools import get_current_datetime, add, compare, count_letter_in_string, search_wikipedia, get_current_temperature

SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""


class Agent:
    def __init__(self, client: OpenAI, model: str, tools: List = [], verbose: bool = True):
        self.client = client
        self.model = model
        self.tools = tools
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的json
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # print(f'{type(tool_call)=}')
        # type(tool_call)=<class 'openai.types.chat.chat_completion_message_function_tool_call.ChatCompletionMessageFunctionToolCall'>
        func_name = tool_call.function.name
        func_args = tool_call.function.arguments
        function_id = tool_call.id
        print(f'{func_name=}  {func_args=}')
        print(f"{func_name} {func_args}  {type(func_args)})") # compare(**{"a": 9.01, "b": 9.1})
        module_name = 'chapter7.Agent.src.tools'
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, func_name)
        except:
            print(f'导入包错误! {module_name=}')
        if callable(func):
            print(f'{func_name=}该函数导入成功!')
        else:
            print(f'{func_name=}该函数导入失败！')

        # 这里func_args是str类型，必须转为dict
        function_call_content = func(**eval(func_args))
        # function_call_content = eval(f"{func_name}(**{func_args})")
        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tools_schema(),
            stream=False
        )
        print(f'{response.choices[0].message.__repr__()=}')
        print(f'{self.get_tools_schema()=}')
        if response.choices[0].message.tool_calls: # 为什么是None?
            print(f'{self.verbose=}')
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print(f'调用工具: {response.choices[0].message.content} {tool_list=}')
            # 再次获取模型的完成响应，这次包含工具调用的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tools_schema(),
                stream=False
            )

        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
