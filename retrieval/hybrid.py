"""
Hybrid Retriever using Reciprocal Rank Fusion (RRF)

Combines:
1. BM25 lexical retrieval
2. Dense semantic retrieval (FAISS + BGE)

Author: Prakhya Khandelwal
"""

import pandas as pd


class HybridRetriever:
    """
    Hybrid Retriever using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    bm25 : BM25Retriever
        BM25 retriever object.

    dense : DenseRetriever
        Dense retriever object.
    """

    def __init__(self, bm25, dense):
        self.bm25 = bm25
        self.dense = dense

    def search(self, query, top_k=50, rrf_k=60):
        """
        Retrieve documents using BM25 + Dense Retrieval
        and combine them using Reciprocal Rank Fusion.

        Parameters
        ----------
        query : str
            User query.

        top_k : int
            Number of candidates retrieved from each retriever.

        rrf_k : int
            Constant used in Reciprocal Rank Fusion.
            Default = 60 (recommended in literature)

        Returns
        -------
        pandas.DataFrame
            Ranked candidate documents.
        """

        # ----------------------------
        # Retrieve candidates
        # ----------------------------

        bm25_results = (
            self.bm25
            .search(query, top_k)
            .reset_index(drop=True)
        )

        dense_results = (
            self.dense
            .search(query, top_k)
            .reset_index(drop=True)
        )

        # ----------------------------
        # RRF Fusion
        # ----------------------------

        fused = {}

        # BM25 contribution
        for rank, row in bm25_results.iterrows():
            doc_id = row["doc_id"]

            if doc_id not in fused:
                fused[doc_id] = row.to_dict()
                fused[doc_id]["rrf_score"] = 0.0

            fused[doc_id]["bm25_score"] = row.get("bm25_score", 0)
            fused[doc_id]["rrf_score"] += 1 / (rrf_k + rank + 1)

        # Dense contribution
        for rank, row in dense_results.iterrows():
            doc_id = row["doc_id"]

            if doc_id not in fused:
                fused[doc_id] = row.to_dict()
                fused[doc_id]["rrf_score"] = 0.0

            fused[doc_id]["dense_score"] = row.get("dense_score", 0)
            fused[doc_id]["rrf_score"] += 1 / (rrf_k + rank + 1)

        # ----------------------------
        # Convert to DataFrame
        # ----------------------------

        merged = pd.DataFrame(fused.values())

        # Fill missing scores
        if "bm25_score" not in merged.columns:
            merged["bm25_score"] = 0.0

        if "dense_score" not in merged.columns:
            merged["dense_score"] = 0.0

        merged["bm25_score"] = merged["bm25_score"].fillna(0)
        merged["dense_score"] = merged["dense_score"].fillna(0)

        # ----------------------------
        # Final ranking
        # ----------------------------

        merged = merged.sort_values(
            "rrf_score",
            ascending=False
        ).reset_index(drop=True)

        return merged

    def debug_search(self, query, top_k=10):
        """
        Helper function to inspect retrieval results.

        Returns only the most useful columns.
        """

        results = self.search(query, top_k=top_k)

        cols = [
            "doc_id",
            "title",
            "source",
            "rrf_score",
            "bm25_score",
            "dense_score",
        ]

        cols = [c for c in cols if c in results.columns]

        return results[cols]