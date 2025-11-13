import json
import os
import re

import PyPDF2
import markdown
import tiktoken
from bs4 import BeautifulSoup

enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    def __init__(self, path: str):
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        files_list = []
        # root, dir, files
        for filepath, dirnames, filenames in os.walk(self._path):
            for filename in filenames:
                if filename.endswith(".md"):
                    files_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    files_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    files_list.append(os.path.join(filepath, filename))

        return files_list

    def get_content(self, max_tokens_len: int = 600, cover_content: int = 150):
        docs = []
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_tokens_len, cover_content=cover_content
            )
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''
        token_len = max_token_len - cover_content
        lines = text.splitlines()

        for line in lines:
            line = line.strip()
            line_len = len(enc.encode(line))
            if line_len > max_token_len:
                if curr_chunk:
                    chunk_text.append(curr_len)
                    curr_chunk = ''
                    curr_len = 0
                # 按照token分割
                line_tokens = enc.encode(line)
                num_chunks = (len(line_tokens) + token_len - 1) // token_len
                for i in range(num_chunks):
                    start_token = i * token_len
                    end_token = min(start_token + token_len, len(line_tokens))

                    chunk_tokens = line_tokens[start_token: end_token]
                    chunk_part = enc.decode(chunk_tokens)

                    # 添加覆盖内容
                    if i > 0 and chunk_text:
                        prev_chunk = chunk_text[-1]
                        cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                        chunk_part = cover_part + chunk_part
                    chunk_text.append(chunk_part)

                curr_chunk = ''
                curr_len = 0
            elif curr_len + line_len + 1 <= token_len:  # +1 for newline
                # 当前行可以加入当前块
                if curr_chunk:
                    curr_chunk += '\n'
                    curr_len += 1
                curr_chunk += line
                curr_len += line_len
            else:
                if curr_chunk:
                    chunk_text.append(curr_chunk)

                # 开始新chunk
                if chunk_text:
                    prev_chunk = chunk_text[-1]
                    cover_part = prev_chunk[-cover_content:] if len(prev_chunk) > cover_content else prev_chunk
                    curr_chunk = cover_part + '\n' + line
                    curr_len += len(enc.encode(cover_part)) + 1 + curr_len
                else:
                    curr_len = line
                    curr_len = line_len

        if curr_chunk:
            chunk_text.append(curr_chunk)
        return chunk_text








    @classmethod
    def read_file_content(cls, file_path: str):
        # 根据文件扩展名选择读取方法
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        with open(file_path, 'r') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path):
        with open(file_path, 'r') as f:
            md_txt = f.read()
            html_text = markdown.markdown(md_txt)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html_parser')
            plain_text = soup.get_text()
            text = re.sub(r'http\S+', '', plain_text)
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class Documents:
    """
        获取已分好类的json格式文档
    """

    def __init__(self, path: str = '') -> None:
        self.path = path

    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content