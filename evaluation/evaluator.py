from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score


class Evaluator:

    def __init__(self):

        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True
        )

    def evaluate(self, prediction, reference):

        # ------------------------
        # ROUGE
        # ------------------------

        rouge = self.rouge.score(
            reference,
            prediction
        )

        # ------------------------
        # BLEU
        # ------------------------

        bleu = sentence_bleu(
            [reference.split()],
            prediction.split(),
            smoothing_function=SmoothingFunction().method1
        )

        # ------------------------
        # BERTScore
        # ------------------------

        P, R, F1 = score(
            [prediction],
            [reference],
            lang="en",
            verbose=False
        )

        return {

            "BLEU": round(bleu,4),

            "ROUGE-1": round(
                rouge["rouge1"].fmeasure,
                4
            ),

            "ROUGE-2": round(
                rouge["rouge2"].fmeasure,
                4
            ),

            "ROUGE-L": round(
                rouge["rougeL"].fmeasure,
                4
            ),

            "BERTScore": round(
                F1.mean().item(),
                4
            )

        }