__all__ = ['AWSEmbeddings']

import os
from typing import Any, List

import requests
from langchain_core.vectorstores import Embeddings


class AWSEmbeddings(Embeddings):
    def __init__(self, api_url: str, api_key: str = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not api_key:
            if 'AWS_API_KEY' in os.environ:
                api_key = os.environ['AWS_API_KEY']
            else:
                raise ValueError('API key is not set')

        self._api_key = api_key
        self._api_url = api_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list()

        for text in texts:
            embeddings.append(self.embed_query(text))

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        request_body = {
            'text': text
        }

        headers = {
            'x-api-key': self._api_key
        }

        response = requests.post(self._api_url, json=request_body, headers=headers).json()
        embedding = response['embedding']
        return embedding
