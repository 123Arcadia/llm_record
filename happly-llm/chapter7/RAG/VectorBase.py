import json
import os.path
from typing import List

import numpy as np
from tqdm import tqdm

from chapter7.RAG.Embeddings import BaseEmbeddings


class VectorStore:
    def __init__(self, document: List[str]=[' '])->None:
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:

        self.vectors = []
        for doc in tqdm(self.document, desc="Caculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = 'storage'):
        """
        持久化
        :param path:
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f'{path}/document.json', 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f'{path}/vectors.json', 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        with open(f'{path}/vectors.json', 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f'{path}/document.json', 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        """
        取最大的k个
        :param query:
        :param EmbeddingModel:
        :param k:
        :return:
        """
        query_v = EmbeddingModel.get_embedding(query)
        res = np.array([self.get_similarity(query_v, vector)  for vector in self.vectors])
        return np.array(self.document)[res.argsort()[-k:][::-1]].tolist()











