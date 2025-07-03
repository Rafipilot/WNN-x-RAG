import ollama
from WeightedRagSystem.Vectorizer import vectorizer
from WeightedRagSystem.ragSystem import ragSystem
from config import openai_key

# Initialize
vec = vectorizer(openai_api_key=openai_key, cache_name="VectorDB.json")
rag = ragSystem(vec)


data = [
    "The Eiffel Tower is located in Paris, France.",
    "Python is a high‑level programming language created by Guido van Rossum.",
    "The Great Wall of China is over 13,000 miles long.",
    "Mount Everest's peak elevation is 8,848 meters above sea level.",
    "The capital of Japan is Tokyo.",
    "Water boils at 100°C at standard atmospheric pressure.",
    "The inventor of the light bulb was Thomas Edison.",
    "The speed of light in vacuum is approximately 299,792 kilometers per second.",
    "The moon orbits Earth approximately every 27.3 days.",
    "The currency of the United Kingdom is the British Pound.",
    "The company made a revenue of $1 million last year.",
    "The weather today is sunny with a chance of rain in the evening.",
    "The stock market saw a significant increase yesterday.",
    "The new product launch was a huge success.",
    "The Eiffel Tower was completed in 1889.",
    "The Empire State Building is located in New York City.",
    "Python is often used for data science and machine learning.",
    "Guido van Rossum named Python after the British comedy group Monty Python.",
    "Thomas Edison also founded General Electric.",
    "Tokyo is one of the most densely populated cities in the world.",
    "The speed of sound is about 343 meters per second in air.",
    "The moon influences Earth's tides.",
    "The British Pound is symbolized as £.",
    "The weather tomorrow will be cloudy with occasional showers.",
    "The stock market experienced a slight drop today.",
    "The recent product launch underperformed due to poor marketing."
]

for snippet in data:
    vec.addToVectorDB(snippet)

test_cases = [
    # ("Where is the Eiffel Tower?", "Eiffel Tower is located in Paris"),
    # ("Who created Python?", "Python is a high‑level programming language created by Guido van Rossum"),
    # ("How long is the Great Wall of China?", "Great Wall of China is over 13,000 miles long"),
    # ("What’s the boiling point of water?", "Water boils at 100°C"),
    ("What’s the capital city of Japan?", "capital of Japan is Tokyo"),
    ("Who invented the light bulb?", "inventor of the light bulb was Thomas Edison"),
    ("Tell me about last year’s revenue.", "revenue of $1 million last year"),
    # ("Will it rain today?", "sunny with a chance of rain in the evening"),
    ("Did stocks go up yesterday?", "significant increase yesterday"),
    #("Was the product launch successful?", "new product launch was a huge success"),
    ### Totally inrellevent info that is not in DB -should output no info found
    ("What is my name", ""), 
    ("What is the capital of the United kingdom", ""), 
    ("What is AO Labs", ""),
    ("Definition of Revenue", ""),
    ("How to make an LLM", ""),
]

def get_rag_feedback(input_text, most_relevant_key):
    msg = [{
        'role': 'user',
        'content': f'Was this information "{most_relevant_key}" mostly generally useful for answering : "{input_text}"? Respond with "yes" or "no".'
    }]
    reply = ollama.chat(model='llama3.2', messages=msg)
    return reply['message']['content'].strip().lower()

def evaluate_rag(test_cases, epochs):
    stats = []
    for epoch in range(1, epochs+1):
        correct = 0
        first_pass = True
        for prompt, expected in test_cases:
            emb = vec.get_embedding(prompt)
            key, min_dist = rag.run_query(emb)


            if first_pass:
                rag.wC.adjust_weights(key)
                first_pass = False
            llm_agrees = False

            if key != "No relevant information found.":
                fb = get_rag_feedback(prompt, key)

                if "yes" in fb.lower():
                    llm_agrees = True
                
                # train weights based on LLM feedback
                
                tag = "neg"
                if llm_agrees:
                    correct += 1
                    tag = "pos"
                    print("Retrival:prompt: ", prompt, " retrived from db: ", key, "llm output: ", fb, "min dist: ", min_dist )
                else:
                    print("retrieval incorrect, prompt: ", prompt, " retrived from db: ", key, "llm output: ", fb, "min dist: ", min_dist)
                    # override = input("Override the llm feedback? ") # this llm is espescially stupid and makes lots of mistakes... consider moving back to openai
                    # # if "yes" in override:
                    # #     correct +=1
                    # #     tag = "pos"
                rag.wC.train_agent(tag, key, rag.ActThresh)

            else:
                print("LLM did not retrieve any info from Vector DB for this prompt: ", prompt)
                # override = input("overide: ")
                # if "yes" in override:
                #     pass
                # else:
                correct += 1 # Rag correctly did not retrieve any data

        accuracy = correct / len(test_cases) * 100
        stats.append((epoch, accuracy))
        print(f"Epoch {epoch}: {accuracy:.1f}% correct retrievals")
    return stats

# Run the evaluation
print("=== Starting RAG evaluation ===")
results = evaluate_rag(test_cases, epochs=20)
print("=== Done ===")
