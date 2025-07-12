# Basic Rag system, looking at the vector database and the input embedding to find the most relevant information.

from sklearn.metrics.pairwise import cosine_similarity  # to calculate distances
import numpy as np
from WeightedRagSystem.weightController import weightController
from WeightedRagSystem.activeThreshold import activeThreshold
        
class ragSystem:
    def __init__(self, vectorizer, activeThresholdTrueFalse=True):
        self.wC = weightController(vectorizer)
        self.vectorizer = vectorizer
        self.vector_db = vectorizer.cache
        self.activeThresholdTrueFalse = activeThresholdTrueFalse

        self.ActThresh = activeThreshold(self.activeThresholdTrueFalse)

        

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

    def run_query(self, input_embedding):
        # 1) Compute weighted distances
        return_array = []
        entries = []
        self.wC.adjust_weights()
        for entry in self.vector_db:
            dist = self.find_distance_embedding(input_embedding, entry["embedding"])
            weighted = dist / entry.get("weight", 1.0)
            return_array.append((entry["input"], weighted))
            entries.append(entry)
        
        return_array.sort(key=lambda x: x[1])
        return_array = return_array[:5]

        if self.activeThresholdTrueFalse:
            filtered = []
            for i, (inp, dist) in enumerate(return_array):
                thresh = self.ActThresh.adjustThreshold(entries[i], input_embedding)
                # print("Current threshold for", inp, ":", thresh)
                if dist <= thresh:
                    filtered.append((inp, dist))
            return_array = filtered

        if not return_array:
            print("No relevant information found")
            return "No relevant information found.", ["No relevant information found."], []

        keys = [inp for inp, i in return_array]
        for entry in self.vector_db:
            if entry["input"] in keys:
                entry["numberOfRetrievals"] = entry.get("numberOfRetrievals", 0) + 1
        
        min_dists = [dist for _, dist in return_array]
        return return_array, keys, min_dists
