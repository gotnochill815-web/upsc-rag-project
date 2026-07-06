import os
import sys
import pandas as pd

PROJECT_ROOT = "/content/drive/MyDrive/upsc_rag_project"

sys.path.insert(0, PROJECT_ROOT)

from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

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
# Build retrievers
# =====================================================

print("Building BM25...")

bm25 = BM25Retriever()
bm25.build(documents)

print("Building Dense...")

dense = DenseRetriever()
dense.build(documents)

hybrid = HybridRetriever(
    bm25,
    dense
)

print("Loading CrossEncoder...")

reranker = CrossEncoderReranker()

# =====================================================
# Load evaluation questions
# =====================================================

questions = pd.read_csv(
    os.path.join(
        PROJECT_ROOT,
        "evaluation",
        "gold_annotations.csv"
    )
)

# =====================================================
# Build Gold Labels
# =====================================================

gold = []

for row in questions.itertuples(index=False):

    print(row.question[:80])

    # Retrieve top 20
    retrieved = hybrid.search(
        row.question,
        top_k=20
    )

    # Rerank
    ranked = reranker.rerank(
        query=row.question,
        documents=retrieved,
        top_k=5
    )

    gold.append(
        ";".join(
            ranked["doc_id"].tolist()
        )
    )

questions["relevant_doc_ids"] = gold

questions.to_csv(
    os.path.join(
        PROJECT_ROOT,
        "evaluation",
        "gold_annotations_final.csv"
    ),
    index=False
)

print("\nDone!")