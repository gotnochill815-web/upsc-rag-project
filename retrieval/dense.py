import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DenseRetriever:

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):

        self.model = SentenceTransformer(model_name)

        self.documents = None
        self.index = None
        self.embeddings = None

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

        results = self.documents.iloc[indices[0]].copy()

        results["dense_score"] = scores[0]

        return results