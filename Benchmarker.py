
from WeightedRagSystem.Vectorizer import vectorizer
from WeightedRagSystem.ragSystem import ragSystem
import numpy as np
from config import openai_key

# Initialize
vec = vectorizer(openai_api_key=openai_key, cache_name="VectorDB.json")
rag = ragSystem(vec)

from datasets import load_dataset

# 2. Load the "covidqa" split from the galileo‑ai/ragbench repo
#    (this gives you a dict with fields ["question","answers","id",…])
dataset = load_dataset("galileo-ai/ragbench", "covidqa", split="train")
print(dataset[0:5])

# 3. Turn it into your (prompt, expected) format.
#    We'll take the first listed answer as the “ground truth.”
test_cases = [
    (ex["question"], ex["answers"][0])
    for ex in dataset
]

# 4. (Optional) If the dataset includes “no‐answer” cases, map those to your
#    "No relevant information found" label instead of answers[0].

# 5. Plug it into your existing evaluation:
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np

def evaluate_system(test_cases, vec, rag, k=5, no_answer_label="No relevant information found"):
    hit1, hitk, mrr = [], [], []
    y_true_cls, y_pred_cls = [], []

    for prompt, expected in test_cases:
        emb = vec.get_embedding(prompt)
        # assume you've extended run_query to return top-k
        keys, dists = rag.run_query(emb, top_k=k)

        # Retrieval metrics
        hit1.append(int(expected in keys[:1]))
        hitk.append(int(expected in keys))
        if expected in keys:
            mrr.append(1.0 / (keys.index(expected) + 1))
        else:
            mrr.append(0.0)

        # No‑answer classification
        true_no = int(expected == no_answer_label)
        pred_no = int(keys[0] == no_answer_label)
        y_true_cls.append(true_no)
        y_pred_cls.append(pred_no)

    print(f"HIT@1: {np.mean(hit1):.3f}")
    print(f"HIT@{k}: {np.mean(hitk):.3f}")
    print(f"MRR:   {np.mean(mrr):.3f}")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_cls, y_pred_cls, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true_cls, y_pred_cls)

    print("\nNo‑Info Classifier:")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Accuracy:  {acc:.3f}")

#evaluate_system(test_cases, vec, rag, k=5)
