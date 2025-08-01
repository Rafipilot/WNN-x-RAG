import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from WeightedRagSystem.Vectorizer import vectorizer
from WeightedRagSystem.ragSystem import ragSystem
from config import openai_key
import random
import numpy as np
from datetime import datetime

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

def compute_metrics(ranks, ks=(1,3), alpha=1, beta=0.25, eps=1e-6, increase_target_weight_amount=5,
                    increase_weight_if_correct=3, decrease_weight_if_incorrect=-1):
    n = len(ranks)
    metrics = {}
    metrics["Number"] = n
    metrics["Alpha"] = alpha
    metrics["Beta"] = beta
    metrics["Epsilon"] = eps
    metrics["Increase Target Weight Amount"] = increase_target_weight_amount
    metrics["Increase Weight If Correct"] = increase_weight_if_correct
    metrics["Decrease Weight If Incorrect"] = decrease_weight_if_incorrect

    # Hit@k
    for k in ks:
        hits = sum(1 for r in ranks if (r is not None and r < k))
        metrics[f"Hit@{k}"] = hits / n
    # MRR
    reciprocal_ranks = [(1.0/(r+1)) if r is not None else 0.0 for r in ranks]
    metrics["MRR"] = sum(reciprocal_ranks) / n
    return metrics

def run_eval(num_trials_array = [30], alpha=[1], beta=[0.25], eps=[1e-6], increase_target_weight_amount=[5], increase_weight_if_correct=[3], decrease_weight_if_incorrect=[-1]):
    metrics_array = []
    #random.shuffle(questions_answers)
    for k, num_trials in enumerate(num_trials_array):
        alpha = alpha[k]
        beta = beta[k]
        eps = eps[k]
        increase_target_weight_amount = increase_target_weight_amount[k]
        increase_weight_if_correct = increase_weight_if_correct[k]
        decrease_weight_if_incorrect = decrease_weight_if_incorrect[k]
        vec = vectorizer(openai_api_key=openai_key, vectorDBName="VectorDB.json")
        rag = ragSystem(vec, activeThresholdTrueFalse=False, alpha=alpha, beta=beta, eps=eps,
                        increase_target_weight_amount=increase_target_weight_amount,
                        increase_weight_if_correct=increase_weight_if_correct,
                        decrease_weight_if_incorrect=decrease_weight_if_incorrect)
        questions_answers =[]
        for ex in dataset.select(range(200)):
            q, a, ctx = ex["question"], ex["answers"]["text"][0], ex["context"]
            questions_answers.append([q, a])
            chunks = sentence_chunker(ctx)
            for chunk in chunks: # tokanization
                vec.addToVectorDB(chunk)
        
        ranks = []

        for i, questions_answer in enumerate(questions_answers[:num_trials]):
            print("Question number:", i)
            now = datetime.now()
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
            #rag.wC.adjust_weights()  # Adjust weights after each training
            print("Time taken for query: ", datetime.now() - now)

            

        metrics = compute_metrics(ranks, alpha=alpha, beta=beta, eps=eps, increase_target_weight_amount=increase_target_weight_amount,
                                  increase_weight_if_correct=increase_weight_if_correct,
                                  decrease_weight_if_incorrect=decrease_weight_if_incorrect)
        metrics_array.append(metrics)
        print(metrics)
        print("finished test number: ", k)
    return metrics_array

if __name__ == "__main__":
    print("Running EVAL")
    metrics_array = run_eval(num_trials_array = [30], alpha=[1], beta=[0.25], eps=[1e-6], increase_target_weight_amount=[5], increase_weight_if_correct=[3], decrease_weight_if_incorrect=[-1])
    print("Finished")
    
    print("Metrics: ", metrics_array)
