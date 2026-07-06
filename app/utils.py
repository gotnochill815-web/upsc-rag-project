import os
import sys

import pandas as pd

# --------------------------------------------------
# Project Root
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Imports
# --------------------------------------------------

from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

from generation.generator import AnswerGenerator


# ==================================================
# Load Retrieval Pipeline
# ==================================================

def load_retriever():

    documents = pd.read_csv(
        os.path.join(
            PROJECT_ROOT,
            "data",
            "processed",
            "documents.csv"
        )
    )

    if "retrieval_text" not in documents.columns:

        documents["retrieval_text"] = (
            documents["title"].fillna("")
            + " "
            + documents["topic"].fillna("")
            + " "
            + documents["text"].fillna("")
        )

    bm25 = BM25Retriever()
    bm25.build(documents)

    dense = DenseRetriever()
    dense.build(documents)

    retriever = HybridRetriever(
        bm25=bm25,
        dense=dense
    )

    reranker = CrossEncoderReranker()

    return retriever, reranker


# ==================================================
# Load Generator
# ==================================================

def load_generator():

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    return AnswerGenerator(model_name)


# ==================================================
# Answer Question
# ==================================================

def answer_question(
    question,
    retriever,
    reranker,
    generator
):

    # Hybrid Retrieval
    retrieved_docs = retriever.search(
        query=question,
        top_k=10
    )

    # CrossEncoder Reranking
    reranked_docs = reranker.rerank(
        query=question,
        documents=retrieved_docs,
        top_k=5
    )

    # Generation
    answer = generator.generate(
        question=question,
        retrieved_docs=reranked_docs
    )

    return answer, reranked_docs