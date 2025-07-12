import nltk
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
from WeightedRagSystem.Vectorizer import vectorizer
from WeightedRagSystem.ragSystem import ragSystem
from config import openai_key

nltk.download('punkt')

vec = vectorizer(openai_api_key=openai_key, cache_name="VectorDB.json")
rag = ragSystem(vec, activeThresholdTrueFalse=False)

dataset = load_dataset("squad", split="validation")

questions, answers = [], []

for ex in dataset.select(range(50)):
    q, a, ctx = ex["question"], ex["answers"]["text"][0], ex["context"]
    questions.append(q)
    answers.append(a)
    for sent in sent_tokenize(ctx):
        vec.addToVectorDB(sent)

rag.wC.adjust_weights()
rag.wC.reset_weights()

correct = 0

for i, (question, answer) in enumerate(zip(questions, answers)):
    emb = vec.get_embedding(question)
    return_array, keys, min_dists = rag.run_query(emb)
    print(f"Query: '{question}' -> Returned keys: {keys}")

    matched_key, matched_dist = None, None
    correct_flag = False
    no_response = True

    if return_array != "No relevant information found.":
        for idx, (key, dist) in enumerate(return_array):
            if answer in key:
                matched_key = key
                matched_dist = dist
                no_response = False
                if not correct_flag:
                    correct += 1
                    correct_flag = True
                print(f"✔ Match found: '{key}' (dist={dist:.4f})")
            else:
                print(f"Training: label=neg, no_response=False, key={key}, dist={dist}")
                rag.wC.train_agent("neg", False, key, dist, idx, rag.ActThresh)
    else:
        no_response = True

    if matched_key and matched_dist:
        print(f"Training: label=pos, no_response={no_response}, key={matched_key}, dist={matched_dist}")
        rag.wC.train_agent("pos", no_response, matched_key, matched_dist, idx, rag.ActThresh)

print(f"Accuracy: {correct / len(questions) * 100:.2f}%")
