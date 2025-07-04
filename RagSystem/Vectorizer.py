
from openai import OpenAI
import openai
import os
import json


class vectorizer:
    def __init__(self, openai_api_key, cache_name="cache.json"):
        openai.api_key = openai_api_key
        global client 
        client = OpenAI(api_key = openai_api_key,)
        self.cache_name = cache_name
        self.load_cache()
        
    def load_cache(self):
        if os.path.exists(self.cache_name):
            with open(self.cache_name, "r") as f:
                self.cache =  json.load(f)
        else:
            self.cache = {}

    def save_cache(self):
        with open(self.cache_name, "w") as f:
            json.dump(self.cache, f)

    def addToVectorDB(self, input):
        if input in self.cache:
            return self.cache[input]
        else:
            response = client.embeddings.create(
                input=input,
                model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            self.cache[input] = embedding
            self.save_cache()
            return embedding
        
    def get_embedding(self, input):
 
        response = client.embeddings.create(
            input=input,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding