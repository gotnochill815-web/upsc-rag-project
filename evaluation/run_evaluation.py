import pandas as pd

from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker
from generation.generator import AnswerGenerator
from evaluation.evaluator import Evaluator


def evaluate_pipeline(
    questions_df,
    hybrid,
    reranker,
    generator
):

    evaluator = Evaluator()

    all_results = []

    for i, row in questions_df.iterrows():

        question = row["question"]

        reference = row["reference_answer"]

        # -----------------------------
        # Retrieval
        # -----------------------------

        retrieved = hybrid.search(
            question
        )

        # -----------------------------
        # Reranking
        # -----------------------------

        top_docs = reranker.rerank(
            question,
            retrieved
        ).head(5)

        # -----------------------------
        # Generation
        # -----------------------------

        prediction = generator.generate(
            question,
            top_docs
        )

        # -----------------------------
        # Evaluation
        # -----------------------------

        metrics = evaluator.evaluate(
            prediction,
            reference
        )

        metrics["question"] = question

        all_results.append(metrics)

        print(f"{i+1}/{len(questions_df)} completed")

    return pd.DataFrame(all_results)