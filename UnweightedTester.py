
import ollama
from WeightedRagSystem.Vectorizer import vectorizer
from WeightedRagSystem import ragSystem
from config import openai_key

vec = vectorizer(openai_api_key=openai_key, cache_name="VectorDB.json")
rag = ragSystem(vec)

dummy_data = [
    "The company made a revenue of $1 million last year.",
    "The weather today is sunny with a chance of rain in the evening.",
    "The stock market saw a significant increase yesterday.",
    "The new product launch was a huge success."
]

for data in dummy_data:
    vec.addToVectorDB(data)

def get_rag_feedback(input_text, most_relevant_key):
    rag_feedback_text = [{
        'role': 'user', 
        'content': f'Was this information "{most_relevant_key}", useful for this prompt: "{input_text}"? Respond with "yes" or "no".'
    }]

    Rag_feedback = ollama.chat(
        model='llama3.2',
        messages=rag_feedback_text
    )

    return Rag_feedback['message']['content']
messages = [
    {'role': 'system', 'content': 'You are a helpful assistant, ensure to use the information from the vector database to answer the question.'},
]

first_pass = True

while True:
    input_text = str(input("Ask a question... "))

    input_embedding = vec.get_embedding(input_text)

    # run a basic Rag query 

    most_relevant_key = rag.run_query(input_embedding)
    print(f"Most relevant key: {most_relevant_key}")

    # Run a prompt using the locally installed LLaMA 3.2 model
    
    messages.append({'role': 'user', 'content': input_text})
    messages.append({'role': 'system', 'content': f'You have access to the following information from the vector database: {most_relevant_key}'})

    response = ollama.chat(
        model='llama3.2',  
        messages=messages,
    )

    messages.append({'role': 'assistant', 'content': response['message']['content']})

    print("Response", response['message']['content'])

    Rag_feedback = get_rag_feedback(input_text, most_relevant_key)

    if first_pass: # If it is the first is the first pass through loop we need to adjust the weights 
        rag.wC.adjust_weights(most_relevant_key)
        first_pass = False

    if most_relevant_key!= "No relevant information found.":
        if "yes" in Rag_feedback.lower():
            print("LLM confirmed the relevance of the information.")
            rag.wC.train_agent("pos", most_relevant_key) # Training the agent with [1,1,1,1]
        else:
            print("LLM did not confirm the relevance of the information.")
            rag.wC.train_agent("neg", most_relevant_key) # Training the agent with [0,0,0,0]
    
