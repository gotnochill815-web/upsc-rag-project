"""
CrossEncoder Reranker

Re-ranks retrieved documents using a CrossEncoder model.

Pipeline:
Hybrid Retrieval
      ↓
CrossEncoder
      ↓
Top-k Documents

Author: Prakhya Khandelwal
"""

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query,
        documents,
        top_k=3
    ):
        """
        Re-rank retrieved documents.

        Parameters
        ----------
        query : str
            User query.

        documents : pandas.DataFrame
            Retrieved documents.

        top_k : int
            Number of documents to keep.

        Returns
        -------
        pandas.DataFrame
            Top reranked documents.
        """

        docs = documents.copy()

        # Build retrieval text if missing
        if "retrieval_text" not in docs.columns:

            docs["retrieval_text"] = (
                docs["title"].fillna("")
                + " "
                + docs["topic"].fillna("")
                + " "
                + docs["text"].fillna("")
            )

        pairs = [
            (query, text)
            for text in docs["retrieval_text"]
        ]

        scores = self.model.predict(pairs)

        docs["rerank_score"] = scores

        docs = docs.sort_values(
            "rerank_score",
            ascending=False
        ).reset_index(drop=True)

        return docs.head(top_k)

    def debug_rerank(
        self,
        query,
        documents,
        top_k=10
    ):

        results = self.rerank(
            query=query,
            documents=documents,
            top_k=top_k
        )

        cols = [
            "doc_id",
            "title",
            "rerank_score"
        ]

        return results[cols]