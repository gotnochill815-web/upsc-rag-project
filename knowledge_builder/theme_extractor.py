import re
from collections import Counter


class ThemeExtractor:

    def __init__(self):

        self.phrases = [

            "Centre-State Relations",
            "President's Rule",
            "Article 356",
            "Discretionary Powers",
            "Reservation of Bills",
            "Ordinance",
            "Aid and Advice",
            "Council of Ministers",
            "Finance Commission",
            "Inter-State Council",
            "Judicial Review",
            "Basic Structure",
            "Fundamental Rights",
            "Directive Principles",
            "Parliament",
            "Lok Sabha",
            "Rajya Sabha",
            "Election Commission",
            "Local Government",
            "Panchayat",
            "Municipality"

        ]

    def extract(self, docs):

        text = " ".join(

            docs["text"]
            .fillna("")
            .astype(str)

        ).lower()

        counter = Counter()

        for phrase in self.phrases:

            count = len(

                re.findall(
                    re.escape(phrase.lower()),
                    text
                )

            )

            if count > 0:

                counter[phrase] = count

        return [

            phrase

            for phrase, _ in counter.most_common()

        ]