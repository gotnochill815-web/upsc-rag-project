import os
import sys
import pandas as pd

PROJECT_ROOT = "/content/drive/MyDrive/upsc_rag_project"

sys.path.insert(0, PROJECT_ROOT)

from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever

from knowledge_builder.collect_documents import DocumentCollector

from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank
)

# =====================================================
# Load corpus
# =====================================================

documents = pd.read_csv(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "documents.csv"
    )
)

# =====================================================
# Build Retrievers
# =====================================================

print("Building BM25...")

bm25 = BM25Retriever()
bm25.build(documents)

print("Building Dense Index...")

dense = DenseRetriever()
dense.build(documents)

retriever = HybridRetriever(
    bm25,
    dense
)

collector = DocumentCollector()

# =====================================================
# Load evaluation questions
# =====================================================

questions = pd.read_csv(
    os.path.join(
        PROJECT_ROOT,
        "evaluation",
        "gold_annotations_final.csv"
    )
)
results = []

# =====================================================
# Evaluation Loop
# =====================================================

for row in questions.itertuples(index=False):

    print(f"Evaluating: {row.question[:70]}...")

    retrieved = retriever.search(
        row.question,
        top_k=10
    )

    retrieved_ids = list(
        retrieved["doc_id"]
    )

    relevant_ids = [
        x.strip()
        for x in row.relevant_doc_ids.split(";")
        if x.strip()
   ]
    results.append({

        "question": row.question,

        "topic": row.knowledge_topic,

        "Recall@5": recall_at_k(
            retrieved_ids,
            relevant_ids,
            5
        ),

        "Recall@10": recall_at_k(
            retrieved_ids,
            relevant_ids,
            10
        ),

        "Precision@5": precision_at_k(
            retrieved_ids,
            relevant_ids,
            5
        ),

        "MRR": reciprocal_rank(
            retrieved_ids,
            relevant_ids
        )

    })

# =====================================================
# Save Results
# =====================================================

results = pd.DataFrame(results)

results.to_csv(

    os.path.join(
        PROJECT_ROOT,
        "evaluation",
        "retrieval_results.csv"
    ),

    index=False

)

print("\n==============================")
print("Average Retrieval Metrics")
print("==============================")

print(results.mean(numeric_only=True))

print("\nSaved to:")

print(
    os.path.join(
        PROJECT_ROOT,
        "evaluation",
        "retrieval_results.csv"
    )
)