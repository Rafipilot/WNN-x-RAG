
import ollama
from RagSystem.Vectorizer import vectorizer
from RagSystem.ragSystem import ragSystem
from config import openai_key

vec = vectorizer(openai_api_key=openai_key, cache_name="VectorDB.json")

dummy_data = [
    "The company made a revenue of $1 million last year.",
    "The weather today is sunny with a chance of rain in the evening.",
    "The stock market saw a significant increase yesterday.",
    "The new product launch was a huge success."
]

for data in dummy_data:
    vec.addToVectorDB(data)

chat_history = []


while True:
    input_text = str(input("Ask a question... "))

    input_embedding = vec.get_embedding(input_text)

    # run a basic Rag query 

    rag = ragSystem()
    most_relevant_key = rag.run_query(input_embedding, vec.cache)
    print(f"Most relevant key: {most_relevant_key}")

    # Run a prompt using the locally installed LLaMA 3.2 model
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'system', 'content': f'You have access to the following information from the vector database: {most_relevant_key}'},
    ]

    for message in chat_history:
        messages.append(message)
    
    messages.append({'role': 'user', 'content': input_text})

    response = ollama.chat(
        model='llama3.2',  
        messages=messages,
    )
    chat_history.append({'role': 'user', 'content': input_text})
    chat_history.append({'role': 'assistant', 'content': response['message']['content']})

    print(response['message']['content'])
