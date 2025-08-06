import random
import numpy as np
from datetime import datetime
import os

from nltk.tokenize import sent_tokenize
from config import openai_key
from datasets import load_dataset

from WeightedRagSystem.Vectorizer import vectorizer
from WeightedRagSystem.ragSystem import ragSystem

# Ensure reproducibility
os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)

# Download punkt if needed
import nltk
nltk.download('punkt')

# Utility: split text into sentence-based chunks

def sentence_chunker(text, chunk_size=300):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ''
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += ' ' + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Benchmark without any training or weight adjustments
def run_regular_rag_benchmark(num_trials=100, vector_db_path='VectorDB_regular.json'):
    # Initialize vectorizer and empty vector DB
    vec = vectorizer(openai_api_key=openai_key, vectorDBName=vector_db_path)

    # Load sample dataset (SQuAD validation)
    dataset = load_dataset('squad', split='validation')
    subset = dataset.select(range(num_trials))

    # Build vector DB from contexts
    print('Building vector DB...')
    for idx, ex in enumerate(subset):
        context = ex['context']
        chunks = sentence_chunker(context)
        for chunk in chunks:
            vec.addToVectorDB(chunk, idx)

    # Initialize RAG system
    rag = ragSystem(vec, activeThresholdTrueFalse=False)
    rag.wC.vector_db_reset()

    # Perform retrieval for each question and record rank
    ranks = []
    print('Running queries...')
    for i, ex in enumerate(subset):
        question = ex['question']
        answer = ex['answers']['text'][0]
        emb = vec.get_embedding(question)
        results, _, _, chunkIDs = rag.run_query(emb)

        # Determine rank of correct chunk
        rank = None
        if isinstance(results, list):
            for idx_res, (text, dist) in enumerate(results):
                if answer in text:
                    rank = idx_res
                    break
        ranks.append(rank)

    # Save DB for future reuse
    vec.save_vectorDB()

    # Compute simple metrics
    from collections import defaultdict
    metrics = defaultdict(int)
    total = len(ranks)
    for r in ranks:
        if r == 0:
            metrics['hit@1'] += 1
        if r is not None and r < 3:
            metrics['hit@3'] += 1
    metrics['hit@1'] /= total
    metrics['hit@3'] /= total

    print(f"Hit@1: {metrics['hit@1']:.3f}")
    print(f"Hit@3: {metrics['hit@3']:.3f}")

    # Compute MRR
    reciprocal_sum = 0.0
    for r in ranks:
        if r is not None:
            reciprocal_sum += 1.0 / (r + 1)
    metrics['MRR'] = reciprocal_sum / total
    print(f"MRR:  {metrics["MRR"]}")
    return metrics

if __name__ == '__main__':
    # Example: benchmark on 100 examples
    metrics = run_regular_rag_benchmark(num_trials=10)
    print('Benchmark metrics:', metrics)
