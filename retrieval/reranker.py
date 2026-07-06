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

        self.model = CrossEncoder(
            model_name
        )

        # Evaluation mode
        if hasattr(self.model, "model"):
            self.model.model.eval()

    # ====================================================
    # Rerank
    # ====================================================

    def rerank(
        self,
        query,
        documents,
        top_k=5
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
            Number of documents returned.

        Returns
        -------
        pandas.DataFrame
            Top reranked documents.
        """

        if documents.empty:
            return documents

        docs = documents.copy()

        # ------------------------------------------------
        # Build retrieval text if missing
        # ------------------------------------------------

        if "retrieval_text" not in docs.columns:

            docs["retrieval_text"] = (

                docs["title"].fillna("")
                + " "
                + docs["topic"].fillna("")
                + " "
                + docs["text"].fillna("")

            )

        # ------------------------------------------------
        # Build query-document pairs
        # ------------------------------------------------

        pairs = [

            (query, text)

            for text in docs["retrieval_text"]

        ]

        # ------------------------------------------------
        # Predict relevance scores
        # ------------------------------------------------

        scores = self.model.predict(
            pairs,
            show_progress_bar=False
        )

        docs["rerank_score"] = scores

        # ------------------------------------------------
        # Sort
        # ------------------------------------------------

        docs = docs.sort_values(
            "rerank_score",
            ascending=False
        ).reset_index(drop=True)

        return docs.head(top_k)

    # ====================================================
    # Debug
    # ====================================================

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
            "source",
            "rerank_score"

        ]

        cols = [
            c for c in cols
            if c in results.columns
        ]

        return results[cols]
