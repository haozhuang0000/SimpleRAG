'''
GPU Server:
Please refer to the set up here: https://github.com/haozhuang0000/RESTAPI_Docker

Local:
Please refer to the set up here: https://github.com/haozhuang0000/NER_News/blob/main/Scripts/VDB_Similarity_Search/Model.py
'''
import os
from typing import Any, Dict, List, Optional
import requests
import json
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class NVEmbed:

    host: str = "http://" + os.getenv('EMBEDDING_HOST') + ":" + os.getenv('EMBEDDING_PORT')
    sub: str = "/api/NVEmbed"
    DIM = 4096

    def __init__(self) -> None:
        super().__init__()
        self.EMBEDDING_API = NVEmbed.host + NVEmbed.sub
        data = {'input': 'test', 'type': 'query'}
        response = requests.post(self.EMBEDDING_API, json=data, verify=False)
        if response.status_code != 200:
            raise ConnectionError(
                'Request failed with status code {}. Please contact the server admin.'.format(response.status_code))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        data = {'input': texts, 'type': 'documents'}
        response = requests.post(self.EMBEDDING_API, json=data, verify=False)
        return response.json()

    def embed_query(self, text: str) -> List[float]:
        data = {'input': text, 'type': 'query'}
        response = requests.post(self.EMBEDDING_API, json=data, verify=False)
        return response.json()