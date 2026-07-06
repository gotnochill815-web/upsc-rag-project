"""
Collect supporting documents for Knowledge Builder.

Author: Prakhya Khandelwal
"""

import os
import yaml
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


class DocumentCollector:

    def __init__(self):

        # -------------------------------------------------
        # Load document corpus
        # -------------------------------------------------

        self.documents = pd.read_csv(
            os.path.join(
                PROJECT_ROOT,
                "data",
                "processed",
                "documents.csv"
            )
        )

        # -------------------------------------------------
        # Load config
        # -------------------------------------------------

        with open(
            os.path.join(
                PROJECT_ROOT,
                "knowledge_builder",
                "config.yaml"
            ),
            "r"
        ) as f:

            self.config = yaml.safe_load(f)

    # =====================================================
    # Constitution Search
    # =====================================================

    def search_constitution(self, article_ids):

        constitution = self.documents[
            self.documents["source"] == "constitution"
        ]

        docs = constitution[
            constitution["doc_id"].isin(article_ids)
        ].copy()

        return docs.reset_index(drop=True)

    # =====================================================
    # PYQ Search
    # =====================================================

    def search_pyqs(self, topics, keywords):

        pyqs = self.documents[
            self.documents["source"] == "pyq"
        ]

        results = pd.DataFrame()

        # ------------------------------------------
        # Topic Match
        # ------------------------------------------

        for topic in topics:

            mask = (

                pyqs["topic"]
                .fillna("")
                .str.contains(
                    topic,
                    case=False,
                    regex=False
                )

            )

            results = pd.concat(
                [results, pyqs[mask]],
                ignore_index=True
            )

        # ------------------------------------------
        # Keyword Match
        # ------------------------------------------

        keyword_results = pd.DataFrame()

        for keyword in keywords:

            mask = (

                pyqs["title"]
                .fillna("")
                .str.contains(
                    keyword,
                    case=False,
                    regex=False
                )

                |

                pyqs["text"]
                .fillna("")
                .str.contains(
                    keyword,
                    case=False,
                    regex=False
                )

            )

            keyword_results = pd.concat(
                [keyword_results, pyqs[mask]],
                ignore_index=True
            )

        results = pd.concat(
            [
                results,
                keyword_results
            ],
            ignore_index=True
        )

        results = results.drop_duplicates(
            subset="doc_id"
        )

        return results.reset_index(drop=True)

    # =====================================================
    # Collect Documents
    # =====================================================

    def collect(self, topic_name):

        topic = self.config["topics"][topic_name]

        constitution_docs = self.search_constitution(

            topic["constitution_articles"]

        )

        pyq_docs = self.search_pyqs(

            topic["pyq_topics"],
            topic["keywords"]

        )

        docs = pd.concat(

            [
                constitution_docs,
                pyq_docs
            ],

            ignore_index=True

        )

        docs = docs.drop_duplicates(
            subset="doc_id"
        )

        docs = docs.reset_index(drop=True)

        return docs

    # =====================================================
    # CrossEncoder Ranking
    # =====================================================

    def rank_documents(self, topic_name, top_k=10):

        from retrieval.reranker import CrossEncoderReranker

        docs = self.collect(topic_name)

        reranker = CrossEncoderReranker()

        topic = self.config["topics"][topic_name]

        query = " ".join(

            [topic_name]

            + topic["keywords"]

        )

        ranked = reranker.rerank(

            query=query,

            documents=docs,

            top_k=top_k

        )

        return ranked.reset_index(drop=True)

    # =====================================================
    # Debug
    # =====================================================

    def debug(self, topic_name):

        docs = self.collect(topic_name)

        return docs[
            [
                "doc_id",
                "source",
                "title"
            ]
        ]