import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from WeightedRagSystem.Vectorizer import vectorizer
from WeightedRagSystem.ragSystem import ragSystem
from config import openai_key
import random
import numpy as np

random.seed(42)
np.random.seed(42)

nltk.download('punkt')

dataset = load_dataset("squad", split="validation")

questions_answers = []

def sentence_chunker(text, chunk_size=300):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks#

def compute_metrics(ranks, ks=(1,3)):
    n = len(ranks)
    metrics = {}
    metrics["Number"] = n
    # Hit@k
    for k in ks:
        hits = sum(1 for r in ranks if (r is not None and r < k))
        metrics[f"Hit@{k}"] = hits / n
    # MRR
    reciprocal_ranks = [(1.0/(r+1)) if r is not None else 0.0 for r in ranks]
    metrics["MRR"] = sum(reciprocal_ranks) / n
    return metrics

def run_eval(num_trials_array = []):
    metrics_array = []
    #random.shuffle(questions_answers)
    for k, num_trials in enumerate(num_trials_array):
        vec = vectorizer(openai_api_key=openai_key, vectorDBName="VectorDB.json")
        rag = ragSystem(vec, activeThresholdTrueFalse=False)
        questions_answers =[]
        for ex in dataset.select(range(200)):
            q, a, ctx = ex["question"], ex["answers"]["text"][0], ex["context"]
            questions_answers.append([q, a])
            chunks = sentence_chunker(ctx)
            for chunk in chunks: # tokanization
                vec.addToVectorDB(chunk)
        
        ranks = []

        for i, questions_answer in enumerate(questions_answers[:num_trials]):
            question = questions_answer[0]
            answer = questions_answer[1]
            emb = vec.get_embedding(question)
            return_array, keys, min_dists = rag.run_query(emb)
            #print(f"Query: '{question}' -> Returned keys: {keys}")

            matched_key, matched_dist, matched_index = None, None, None
            correct_flag = False
            no_response = True

            if return_array != "No relevant information found.":
                for idx, (key, dist) in enumerate(return_array):
                    if answer in key:
                        matched_key = key
                        matched_dist = dist
                        no_response = False
                        if not correct_flag:
                            correct_flag = True
                            matched_index = idx
                            ranks.append(idx)
                    # print(f"✔ Match found: '{key}' (dist={dist:.4f})")
                    else:                
                        if (idx ==0 or idx ==1) and dist < 0.35: # if it is top 1 or 2 and incorrect then the weight is too large
                            print(f"Training: label=neg, no_response=False, key={key}, dist={dist}")
                            rag.wC.train_agent("neg", False, key, dist, idx, rag.ActThresh)   

            if matched_key and matched_dist:
                print(f"Training: label=pos, no_response=False, key={matched_key}, dist={matched_dist}")
                rag.wC.train_agent("pos", False, matched_key, matched_dist, matched_index, rag.ActThresh)
            else:
                print("Faliure of RAG sys query: ", question, " ra: ", return_array, " answer: ", answer)
                #rag.wC.train_agent("neg", True, matched_key, matched_dist, matched_index, rag.ActThresh)
                rag.wC.increase_target_weight(answer) # Increase the weight of the expected retrieval in the vector DB
                ranks.append(None)


            print("Question number:", i)

        metrics = compute_metrics(ranks)
        metrics_array.append(metrics)
        print(metrics)
        print("finished test number: ", k)
    return metrics_array

if __name__ == "__main__":
    print("Running EVAL")
    metrics_array = run_eval(num_trials_array=[30,60,90,120])
    print("Finished")
    
    print("Metrics: ", metrics_array)
