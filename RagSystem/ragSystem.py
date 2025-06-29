# Basic Rag system, looking at the vector database and the input embedding to find the most relevant information.

from openai import OpenAI
import openai
import os
import json
from sklearn.metrics.pairwise import cosine_similarity  # to calculate distances
import numpy as np

        
class ragSystem:
    def __init__(self):
        pass

    def normalize(self, embedding): 
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding

    def find_distance_embedding(self, input_embedding, embedding):
        input_embedding = self.normalize(input_embedding)
        embedding = self.normalize(embedding)

        input_embedding = input_embedding.reshape(1, -1)  # Reshape for sklearn
        embedding = embedding.reshape(1, -1)

        similarities = cosine_similarity(input_embedding, embedding)
        distances = 1 - similarities  # Convert similarity to distance

        return distances[0][0]  # Return the distance value

    def run_query(self, input_embedding, vector_db):
        # This will loop through the vector database at every datapoint and use a cosine similarity to find the most relevent information.
        min_dist = float("inf")  # Initialize to a large value
        most_relevant_key = None
        for entry in vector_db:
            key = entry["input"]
            embedding = entry["embedding"]
            distance = self.find_distance_embedding(input_embedding, embedding)
            if distance < min_dist:
                min_dist = distance
                most_relevant_key = key
        if min_dist > 0.2:
            print("No relevant information found, min distance:", min_dist)
            return "No relevant information found."

        return most_relevant_key
 