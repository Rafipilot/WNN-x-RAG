# Basic Rag system, looking at the vector database and the input embedding to find the most relevant information.


from sklearn.metrics.pairwise import cosine_similarity  # to calculate distances
import numpy as np
from RagSystem.weightController import weightController  # to adjust weights in the vector database
        
class ragSystem:
    def __init__(self):
        self.wC = weightController()

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

    def run_query(self, input_embedding, vector_db, vectorizer):
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
        
        for entry in vector_db:
            if entry["input"] == most_relevant_key:
                entry["numberOfRetrievals"] += 1

        self.adjust_weights(vector_db, vectorizer)
        return most_relevant_key
    
    def adjust_weights(self, vector_db, vectorizer):
        self.wC.adjust_weights(vector_db, vectorizer)
 