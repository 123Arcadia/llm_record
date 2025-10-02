import os

import numpy as np
import tiktoken
from dotenv import load_dotenv, find_dotenv
from numpy import dtype


def test_env():
    # o = load_dotenv(find_dotenv())
    o = load_dotenv(dotenv_path='../.env_examples')

    print(f'{o=}')
    print(f'{os.getenv("OPENAI_API_KEY")}')
    print(f'{os.getenv("OPENAI_BASE_URL")}')


def test_findenv():
    o = find_dotenv('../.env_examples')
    print(f'{o=}')
    # o='/home/zhangchenwei/llm_record/chapter7/RAG/.env_examples'


def test_topk():
    np.random.seed(42)
    a = np.random.randint(1,10, size=(1,10))
    print(f'{a=}')
    print(f'{a.argsort()=}')
    k = 3
    print(f'{a.argsort()[-k:]=}')
    print(f'{a.argsort()[-k:][::-1]=}')

def test_get_files():
    list = os.walk('./')
    for l in list:
        print(l)
        # ('./', ['.pytest_cache', '__pycache__'], ['.env_examples', 'utils.py', 'LLM.py', 'demo.py', 'test_dotenv.py', 'Embeddings.py', 'VectorBase.py'])
        # ('./.pytest_cache', ['v'], ['.gitignore', 'CACHEDIR.TAG', 'README.md'])
        # ('./.pytest_cache/v', ['cache'], [])
        # ('./.pytest_cache/v/cache', [], ['stepwise', 'lastfailed', 'nodeids'])
        # ('./__pycache__', [], ['test_dotenv.cpython-38-pytest-8.3.5.pyc'])

def test_tiktoken():
    enc = tiktoken.get_encoding("cl100k_base")
    print(f'{enc.__dict__.keys()=}')
    # enc.__dict__.keys()=dict_keys(['name', '_pat_str', '_mergeable_ranks', '_special_tokens', 'max_token_value', '_core_bpe'])
    print(f'{enc.__dict__["name"]=}')
    print(f'{enc.__dict__["max_token_value"]=}')
    # enc.__dict__["name"]='cl100k_base'
    # enc.__dict__["max_token_value"]=100276
    print(f'{enc.__dict__["_core_bpe"]=}')


def test_split_chunk():
    a = 3
    b = 13
    chunk_nums = (b + a - 1) // a
    print(f'{chunk_nums=}') # a4 b13 ->3
    print(f'{chunk_nums=}') # a2 b13 ->7
    print(f'{chunk_nums=}') # a3 b13 ->5