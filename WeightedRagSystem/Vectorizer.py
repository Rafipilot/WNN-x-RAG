
from openai import OpenAI
import openai
import os
import json


class vectorizer:
    def __init__(self, openai_api_key, vectorDBName="VectorDB.json", cache_name="cache.json"):
        openai.api_key = openai_api_key
        global client 
        client = OpenAI(api_key = openai_api_key,)
        self.vectorDBName = vectorDBName
        self.cache_name = cache_name
        self.load_VectorDB()
        
    def load_VectorDB(self):
        if os.path.exists(self.vectorDBName):
            with open(self.vectorDBName, "r") as f:
                self.vectorDB =  json.load(f)
        else:
            self.vectorDB = []
        if os.path.exist(self.cache_name):
            with open(self.cache_name, "r") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def save_vectorDB(self):
        try:
            with open(self.vectorDBName, "w") as f:
                json.dump(self.vectorDB, f)
            with open(self.cache_name, "w") as f:
                json.dump(self.cache, f)
        except Exception as e:
            print("error: ", e)

    def addToVectorDB(self, input):
        for entry in self.vectorDB:
            if entry["input"] == input:
                return entry["embedding"]
        
        print("adding to vector DB")
        embedding = self.get_embedding(input)
        # Save the embedding to the vector db
        new_entry = {
            "input": input,
            "embedding": embedding,
            "weight": 0.8,  # A placeholder for the iconic trainable weight
            "numberOfRetrievals": 0,
            "numberFailures": 0,
            "uniqueID": len(self.vectorDB) + 1  # Unique identifier for the entry
        }
        self.vectorDB.append(new_entry)
        self.save_vectorDB()
        return embedding
    
    def addToCache(self, input, embedding):
        for entry in self.cache:
            if entry["input"] == input:
                return entry["embedding"]
        self.cache[input] = embedding
    
    def incrementNumberFailures(self, input):
        for i,entry in enumerate(self.vectorDB):
            if entry["input"] == input:
                self.vectorDB[i]["numberFailures"] += 1
                break
        self.save_vectorDB()
        
    def get_embedding(self, input):  # to do introduce a caching mechanism for embeddings not in vector db for eval

        for entry in self.cache:
            if entry == input:
                return self.cache[entry]
        
        response = client.embeddings.create(
            input=input,
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        self.addToCache(input, embedding)
        print("Adding to cache")
        return embedding