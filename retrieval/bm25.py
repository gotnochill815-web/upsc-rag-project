from rank_bm25 import BM25Okapi
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class BM25Retriever:

    def __init__(self):

        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()

        self.documents = None
        self.bm25 = None

    def preprocess(self, text):

        text = text.lower()

        words = re.findall(r"[a-zA-Z0-9]+", text)

        words = [
            w for w in words
            if w not in self.stop_words
        ]

        words = [
            self.stemmer.stem(w)
            for w in words
        ]

        return words

    def build(self, documents):

        self.documents = documents

        corpus = [
            self.preprocess(doc)
            for doc in documents["retrieval_text"]
        ]

        self.bm25 = BM25Okapi(corpus)

    def search(self, query, top_k=20):

        scores = self.bm25.get_scores(
            self.preprocess(query)
        )

        results = self.documents.copy()

        results["bm25_score"] = scores

        return results.nlargest(
            top_k,
            "bm25_score"
        )