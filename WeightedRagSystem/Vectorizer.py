
from openai import OpenAI
import openai
import os
import json


class vectorizer:
    def __init__(self, openai_api_key, cache_name="VectorDB.json"):
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
            self.cache = []

    def save_cache(self):
        try:
            with open(self.cache_name, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print("error: ", e)

    def addToVectorDB(self, input):
        for entry in self.cache:
            if entry["input"] == input:
                return entry["embedding"]
            

        print("adding to vector DB")
        embedding = self.get_embedding(input)
        # Save the embedding to the cache
        new_entry = {
            "input": input,
            "embedding": embedding,
            "weight": 0.8,  # A placeholder for the iconic trainable weight
            "numberOfRetrievals": 0,
            "numberFailures": 0,
            "uniqueID": len(self.cache) + 1  # Unique identifier for the entry
        }
        self.cache.append(new_entry)
        self.save_cache()
        return embedding
    
    def incrementNumberFailures(self, input):
        for i,entry in enumerate(self.cache):
            if entry["input"] == input:
                self.cache[i]["numberFailures"] += 1
                print("Incrementing num faliures")
                break
        self.save_cache()
        
    def get_embedding(self, input):
 
        response = client.embeddings.create(
            input=input,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        return embedding