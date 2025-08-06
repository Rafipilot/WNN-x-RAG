from nltk.tokenize import sent_tokenize
import nltk
from config import openai_key

import random
import numpy as np
from datetime import datetime
import os

from datasets import load_dataset
os.environ["PYTHONHASHSEED"] = "0"

random.seed(42)
np.random.seed(42)

nltk.download('punkt')




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

#vec._save_snapshot()  # Save the initial state of the vector DB

def run_eval(
    num_trials=30,
    alpha=1,
    beta=0.25,
    eps=1e-6,
    increase_target_weight_amount=5,
    increase_weight_if_correct=3,
    decrease_weight_if_incorrect=-1,
):
    from WeightedRagSystem.Vectorizer import vectorizer
    from WeightedRagSystem.ragSystem import ragSystem
    
    import random
    import numpy as np
    from datetime import datetime
    import os

    from datasets import load_dataset
    os.environ["PYTHONHASHSEED"] = "0"

    random.seed(42)
    np.random.seed(42)
    vec = vectorizer(openai_api_key=openai_key, vectorDBName="VectorDB.json")
    questions_answers =[]
    dataset = load_dataset("squad", split="validation")

    previous_ctx=None
    chunkID=0
    subset = dataset.select(range(1000))

    groups = []
    current_ctx = None
    current_group = []
    for ex in subset:
        ctx = ex["context"]
        if ctx != current_ctx:

            if current_group:
                groups.append(current_group)
            # Start new group
            current_group = [ex]
            current_ctx = ctx
        else:
            current_group.append(ex)

    if current_group:
        groups.append(current_group)

    random.shuffle(groups)

    questions_answers = []
    chunkID = 0
    for group in groups[:5]:

        for ex in group:
            q = ex["question"]
            a = ex["answers"]["text"][0]
            questions_answers.append([q, a])


        chunkID += 1
    
        shared_ctx = group[0]["context"]
        chunks = sentence_chunker(shared_ctx)
        for chunk in chunks:
            vec.addToVectorDB(chunk, chunkID)

    
    #vec = vectorizer(openai_api_key=openai_key, vectorDBName="VectorDB.json") 

    rag = ragSystem(vec, activeThresholdTrueFalse=False, alpha=alpha, beta=beta, eps=eps,
                    increase_target_weight_amount=increase_target_weight_amount,
                    increase_weight_if_correct=increase_weight_if_correct,
                    decrease_weight_if_incorrect=decrease_weight_if_incorrect) # re init  rag sys
    
    rag.wC.vector_db_reset() # There is no other info stored in vectorDB so this resets all of it, so exact equivalent of making new one 


    ranks = []
    for i, questions_answer in enumerate(questions_answers[:num_trials]):
        print("Question number:", i)

        now = datetime.now()
        question = questions_answer[0]
        answer = questions_answer[1]
        emb = vec.get_embedding(question) # this rerieves the embedding from a cache generally
        
        return_array, keys, min_dists, chunkIDs = rag.run_query(emb)
        #print(f"Query: '{question}' -> Returned keys: {keys}")

        matched_key, matched_dist, matched_index = None, None, None
        correct_flag = False
        no_response = True

        if return_array != "No relevant information found.":

            for idx, (key, dist) in enumerate(return_array[:3]):
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
                        pass
                        #print(f"Training: label=neg, no_response=False, key={key}, dist={dist}")
                        rag.wC.train_agent("neg", False, key, dist, idx, rag.ActThresh)   

        if matched_key and matched_dist:
            #print(f"Training: label=pos, no_response=False, key={matched_key}, dist={matched_dist}")
            rag.wC.train_agent("pos", False, matched_key, matched_dist, matched_index, rag.ActThresh)
        else:
            #print("Faliure of RAG sys query: ", question, " ra: ", return_array, " answer: ", answer)
            #rag.wC.train_agent("neg", True, matched_key, matched_dist, matched_index, rag.ActThresh)
            
            rag.wC.increase_target_weight(answer) # Increase the weight of the expected retrieval in the vector DB
            ranks.append(None)
        rag.wC.adjust_weights(chunkIDs)  # Adjust weights after each training
        print("Time taken for query: ", datetime.now() - now)
        
    vec.save_vectorDB()# save db once at the end
    metrics = compute_metrics(ranks, alpha=alpha, beta=beta, eps=eps, increase_target_weight_amount=increase_target_weight_amount,
                                increase_weight_if_correct=increase_weight_if_correct,
                                decrease_weight_if_incorrect=decrease_weight_if_incorrect)

    return metrics

def run_full_eval(num_trials_array, alpha=1, beta=0.2, eps=1e-6, increase_target_weight_amount=5, increase_weight_if_correct=3, decrease_weight_if_incorrect=-1):
    metrics_array = []
    for k,num_trials in enumerate(num_trials_array):
        α = alpha[k] if isinstance(alpha, list) else alpha
        β = beta[k]   if isinstance(beta, list)  else beta
        ε = eps[k]    if isinstance(eps, list)   else eps
        inc_tgt = (increase_target_weight_amount[k]
                   if isinstance(increase_target_weight_amount, list)
                   else increase_target_weight_amount)
        inc_corr = (increase_weight_if_correct[k]
                    if isinstance(increase_weight_if_correct, list)
                    else increase_weight_if_correct)
        dec_incorr = (decrease_weight_if_incorrect[k]
                      if isinstance(decrease_weight_if_incorrect, list)
                      else decrease_weight_if_incorrect)

        print(f"\n=== Run {k+1}: α={α}, β={β}, ε={ε}, trials={num_trials} ===")
        metrics = run_eval(num_trials, alpha=α, beta=β, eps=ε,
                                 increase_target_weight_amount=inc_tgt,
                                 increase_weight_if_correct=inc_corr,
                                 decrease_weight_if_incorrect=dec_incorr)
        print(f"Metrics for run {k+1}: {metrics}")
        metrics_array.append(metrics)
    return metrics_array

if __name__ == "__main__":
    print("Running EVAL")
    # alpha_values = [1.0, 0.975, 0.95]
    # beta_values  = [0.22,	0.21,	0.20,	0.19,	0.18]

    # # Number of trials per combination
    # num_trials_array = [120] * (len(alpha_values) * len(beta_values))

    # # Create full alpha-beta grid
    # alphas = [a for a in alpha_values for _ in beta_values]
    # betas  = [b for _ in alpha_values for b in beta_values]

    # Run evaluation
    metrics_array = run_full_eval(
        num_trials_array=[120],
        # increase_target_weight_amount=5,
        # increase_weight_if_correct=3,
        # decrease_weight_if_incorrect=-1
    )

    print("Finished")
    
    print("Metrics: ", metrics_array)
