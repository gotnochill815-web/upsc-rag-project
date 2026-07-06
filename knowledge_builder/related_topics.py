import re


class RelatedTopicFinder:

    def __init__(self, config):

        self.topics = list(config["topics"].keys())

    def find(self, docs, current_topic):

        text = " ".join(
            docs["text"]
            .fillna("")
            .astype(str)
        ).lower()

        related = []

        for topic in self.topics:

            if topic == current_topic:
                continue

            if topic.lower() in text:
                related.append(topic)

        return sorted(related)