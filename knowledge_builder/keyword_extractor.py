import re
from collections import Counter


class KeywordExtractor:

    def __init__(self):

        self.stopwords = {

            "the","and","of","to","for","in","on",
            "a","an","is","are","be","shall",
            "state","states","india","indian",
            "article","constitution","government"

        }

    def extract(self, docs, top_k=20):

        text = " ".join(

            docs["text"]
            .fillna("")
            .astype(str)

        )

        words = re.findall(
            r"[A-Za-z][A-Za-z\-]+",
            text
        )

        words = [

            w

            for w in words

            if len(w) > 3
            and w.lower() not in self.stopwords

        ]

        counter = Counter(words)

        return [

            word

            for word, _ in counter.most_common(top_k)

        ]