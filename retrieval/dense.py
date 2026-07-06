import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(
        self,
        model_name="BAAI/bge-base-en-v1.5",
        index_dir="indexes"
    ):

        self.model = SentenceTransformer(model_name)

        self.documents = None
        self.index = None
        self.embeddings = None

        self.index_dir = index_dir

        os.makedirs(
            self.index_dir,
            exist_ok=True
        )

    # ====================================================
    # Build Index
    # ====================================================

    def build(self, documents):

        self.documents = documents

        texts = documents["retrieval_text"].fillna("").tolist()

        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        ).astype(np.float32)

        self.index = faiss.IndexFlatIP(
            self.embeddings.shape[1]
        )

        self.index.add(self.embeddings)

    # ====================================================
    # Save
    # ====================================================

    def save(self):

        faiss.write_index(
            self.index,
            os.path.join(
                self.index_dir,
                "dense.index"
            )
        )

        np.save(
            os.path.join(
                self.index_dir,
                "dense_embeddings.npy"
            ),
            self.embeddings
        )

    # ====================================================
    # Load
    # ====================================================

    def load(self, documents):

        self.documents = documents

        self.index = faiss.read_index(

            os.path.join(
                self.index_dir,
                "dense.index"
            )

        )

        self.embeddings = np.load(

            os.path.join(
                self.index_dir,
                "dense_embeddings.npy"
            )

        )

    # ====================================================
    # Search
    # ====================================================

    def search(self, query, top_k=20):

        query_embedding = self.model.encode(

            [query],

            normalize_embeddings=True,

            convert_to_numpy=True

        ).astype(np.float32)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = self.documents.iloc[
            indices[0]
        ].copy()

        results["dense_score"] = scores[0]

        return results
