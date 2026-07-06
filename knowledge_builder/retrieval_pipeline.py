"""
Knowledge Retrieval Pipeline

Uses the same retrieval stack as the RAG system.

Author: Prakhya Khandelwal
"""

import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.insert(0, PROJECT_ROOT)

from retrieval.bm25 import BM25Retriever
from retrieval.dense import DenseRetriever
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker


class KnowledgeRetriever:

    def __init__(self, documents):

        self.documents = documents.copy()

        if "retrieval_text" not in self.documents.columns:

            self.documents["retrieval_text"] = (
                self.documents["title"].fillna("")
                + " "
                + self.documents["topic"].fillna("")
                + " "
                + self.documents["text"].fillna("")
            )

        self.bm25 = BM25Retriever()
        self.bm25.build(self.documents)

        self.dense = DenseRetriever()
        self.dense.build(self.documents)

        self.hybrid = HybridRetriever(
            bm25=self.bm25,
            dense=self.dense
        )

        self.reranker = CrossEncoderReranker()

    def retrieve(self, query, top_k=10):

        docs = self.hybrid.search(
            query=query,
            top_k=30
        )

        docs = self.reranker.rerank(
            query=query,
            documents=docs,
            top_k=top_k
        )

        return docs.reset_index(drop=True)