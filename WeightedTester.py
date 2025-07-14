
from WeightedRagSystem.Vectorizer import vectorizer
from WeightedRagSystem.ragSystem import ragSystem
import numpy as np
from config import openai_key

# Initialize
vec = vectorizer(openai_api_key=openai_key, cache_name="VectorDB.json")
rag = ragSystem(vec)
rag.wC.reset_weights()

# data = [
#     "The Eiffel Tower is located in Paris, France.",
#     "Python is a high‑level programming language created by Guido van Rossum.",
#     "The Great Wall of China is over 13,000 miles long.",
#     "Mount Everest's peak elevation is 8,848 meters above sea level.",
#     "The capital of Japan is Tokyo.",
#     "Water boils at 100°C at standard atmospheric pressure.",
#     "The inventor of the light bulb was Thomas Edison.",
#     "The speed of light in vacuum is approximately 299,792 kilometers per second.",
#     "The moon orbits Earth approximately every 27.3 days.",
#     "The currency of the United Kingdom is the British Pound.",
#     "The company made a revenue of $1 million last year.",
#     "The weather today will rain in the evening.",
#     "The stock market saw a significant increase yesterday.",
#     "The new product launch was a huge success.",
#     "The Eiffel Tower was completed in 1889.",
#     "The Empire State Building is located in New York City.",
#     "Python is often used for data science and machine learning.",
#     "Guido van Rossum named Python after the British comedy group Monty Python.",
#     "Thomas Edison also founded General Electric.",
#     "Tokyo is one of the most densely populated cities in the world.",
#     "The speed of sound is about 343 meters per second in air.",
#     "The moon influences Earth's tides.",
#     "The British Pound is symbolized as £.",
#     "The weather tomorrow will be cloudy with occasional showers.",
#     "The stock market experienced a slight drop today.",
# ]

# for snippet in data:
#     vec.addToVectorDB(snippet)

# train_cases = [
#     ("Where is the Eiffel Tower?", "Eiffel Tower is located in Paris"),
#     ("Who created Python?", "Guido van Rossum"),
#     ("How long is the Great Wall of China?", "Great Wall of China is over 13,000 miles long"),
#     ("What’s the boiling point of water?", "Water boils at 100°C"),
#     ("What’s the capital city of Japan?", "capital of Japan is Tokyo"),
#     ("Who invented the light bulb?", "inventor of the light bulb was Thomas Edison"),
#     ("Tell me about last year’s revenue.", "revenue of $1 million last year"),
#     ("Will it rain today?", "will rain in the evening"),
#     ("Did stocks go up yesterday?", "significant increase yesterday"),
#     ("Was the product launch successful?", "new product launch was a huge success"),
#     ### Totally inrellevent info that is not in DB -should output no info found
#     ("What is my name", "No relevant information found"), 
#     ("What is the capital of the United kingdom", "No relevant information found"), 
#     ("What is AO Labs", "No relevant information found"),
#     ("How to make an LLM", "No relevant information found"),
# ]


# test_cases= [
#         ("what is the speed of sound","about 343 meters per second"),
#         ("Who founded General Electric?", "Thomas Edison also founded General Electric"),
#         ("How long does the moon take to orbit Earth?", "moon orbits Earth approximately every 27.3 days"),
#         ("What affects Earth's tides?", "moon influences Earth's tides"),
#         ("What is the currency of the UK?", "currency of the United Kingdom is the British Pound"),
#         ("What symbol represents the British Pound?", "British Pound is symbolized as £"),
#         ("When was the Eiffel Tower completed?", "Eiffel Tower was completed in 1889"),
#         ("Where is the Empire State Building?", "Empire State Building is located in New York City"),
#         ("Who named Python after Monty Python?", "Guido van Rossum named Python after the British comedy group Monty Python"),
#         ("What happened to the stock market today?", "stock market experienced a slight drop today"),
#         ("What is the population of Canada?", "No relevant information found"),
#         ("How do black holes form?", "No relevant information found"),
#         ("What is the tallest building in the world?", "No relevant information found"),
#         ("What is 2+2?", "No relevant information found"),
#         ("Tell me about the Amazon rainforest", "No relevant information found"),
#         ("What is the GDP of India?", "No relevant information found"),
#         ("When did World War II end?", "No relevant information found"),
#         ("What is a quantum computer?", "No relevant information found"),
#         ("How many moons does Mars have?", "No relevant information found"),
#         ("Explain relativity theory", "No relevant information found"),

# ]

# # def get_rag_feedback(input_text, most_relevant_key):
# #     msg = [{
# #         'role': 'user',
# #         'content': f'Was this information "{most_relevant_key}" mostly generally useful for answering : "{input_text}"?. be quite careful since your response is being used to train the rag system; a fault could mess it up. Respond with "yes" or "no".'
# #     }]
# #     reply = ollama.chat(model='llama3.2', messages=msg)
# #     return reply['message']['content'].strip().lower()

# def train_rag(train_cases, epochs=3):

#     for epoch in range(epochs):
#         correct = 0
#         # Adjust weights once at start of epoch
#         rag.wC.adjust_weights()

#         for prompt, expected in train_cases:
#             emb = vec.get_embedding(prompt)
#             return_array, keys, min_dists = rag.run_query(emb)
#             print(f"Query: '{prompt}' -> Returned keys: {keys}")

#             # Determine if any key matches expected substring in top-3 results
#             no_response = True
#             matched_key = None
#             matched_dist = None

#             # Check top-3 candidates
#             index = None
#             if return_array != "No relevant information found.":
#                 for i, (key, dist) in enumerate(return_array[:3]):
#                     if expected in key:
#                         matched_key = key
#                         matched_dist = dist
#                         no_response = False
#                         correct += 1
#                         index =i
#                         print(f"✔ Match found: '{key}' (dist={dist:.4f})")
#                     else:
#                         no_response = False
#                         label = "neg"
                        
#                         print(f"Training: label={label}, no_response={no_response}, key={key}, dist={dist}")
#                         rag.wC.train_agent(label, no_response, key, dist, i, rag.ActThresh)
            

#             else:
#                 no_response = True

#                                 # Define training label
#             if matched_key:
#                 label = "pos"
#             else:
#                 label = "neg"

#             # If system returned nothing at all but expected 'No relevant information found'
#             if not keys and expected == "No relevant information found":
#                 no_response = True
#                 label = "pos"
#                 correct += 1
#                 print("✔ Correct no-response")

#             # Train agent on this instance
#             if matched_key and matched_dist:
#                 print(f"Training: label={label}, no_response={no_response}, key={matched_key}, dist={matched_dist}")
#                 rag.wC.train_agent(label, no_response, matched_key, matched_dist,index, rag.ActThresh)
            

#         accuracy = (correct / len(train_cases)) * 100
#         print(f"Epoch {epoch}/{epochs}: {accuracy:.1f}% accuracy")


# def Test_rag(test):
#         correct = 0
#         # Adjust weights once at start of epoch
#         rag.wC.adjust_weights()

#         for prompt, expected in test:
#             emb = vec.get_embedding(prompt)
#             return_array, keys, min_dists = rag.run_query(emb)
#             print(f"Query: '{prompt}' -> Returned keys: {keys}")

#             # Determine if any key matches expected substring in top-3 results
#             no_response = True
#             matched_key = None
#             matched_dist = None

#             # Check top-3 candidates
#             index = None
#             if return_array != "No relevant information found.":
#                 for i, (key, dist) in enumerate(return_array[:3]):
#                     if expected in key:
#                         matched_key = key
#                         matched_dist = dist
#                         no_response = False
#                         correct += 1
#                         index =i
#                         print(f"✔ Match found: '{key}' (dist={dist:.4f})")
#                     else:
#                         no_response = False
#                         label = "neg"
                        
#                         #print(f"Training: label={label}, no_response={no_response}, key={key}, dist={dist}")
#                         # rag.wC.train_agent(label, no_response, key, dist, i, rag.ActThresh)
            

#             else:
#                 no_response = True

#                                 # Define training label
#             if matched_key:
#                 label = "pos"
#             else:
#                 label = "neg"

#             # If system returned nothing at all but expected 'No relevant information found'
#             if not keys and expected == "No relevant information found":
#                 no_response = True
#                 label = "pos"
#                 correct += 1
#                 print("✔ Correct no-response")

#             # Train agent on this instance
#             if matched_key and matched_dist:
#                 print(f"Training: label={label}, no_response={no_response}, key={matched_key}, dist={matched_dist}")
#                 rag.wC.train_agent(label, no_response, matched_key, matched_dist,index, rag.ActThresh)
            
#         accuracy = (correct / len(test)) * 100
#         print("acc: ", accuracy)

# # Run the evaluation
# print("=== Starting RAG Training ===")
# results = train_rag(train_cases, epochs=1)
# print("=== Done ===")

# # print(" === Starting Testing ===")
# Test_rag(test_cases)
# # print("=== Done ===")
# # while True:
# #     user_input = input("Ask... ")
# #     user_embedding = vec.get_embedding(user_input)

# #     key, min_dist = rag.run_query(user_embedding)

# #     print("Recieved key: ", key)
