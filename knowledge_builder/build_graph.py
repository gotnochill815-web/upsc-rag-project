import os
import networkx as nx

from pyvis.network import Network

from collect_documents import DocumentCollector
from keyword_extractor import KeywordExtractor
from theme_extractor import ThemeExtractor
from related_topics import RelatedTopicFinder

PROJECT_ROOT = "/content/drive/MyDrive/upsc_rag_project"

collector = DocumentCollector()

theme_extractor = ThemeExtractor()
related_finder = RelatedTopicFinder(
    collector.config
)

graph = nx.Graph()

for topic in collector.config["topics"]:

    docs = collector.collect(topic)

    graph.add_node(
        topic,
        group="topic"
    )

    # ------------------------
    # Constitution Articles
    # ------------------------

    constitution = docs[
        docs.source == "constitution"
    ]

    for row in constitution.itertuples():

        article = row.doc_id

        graph.add_node(
            article,
            group="article"
        )

        graph.add_edge(
            topic,
            article
        )

    # ------------------------
    # Themes
    # ------------------------

    themes = theme_extractor.extract(docs)

    for theme in themes:

        graph.add_node(
            theme,
            group="theme"
        )

        graph.add_edge(
            topic,
            theme
        )

    # ------------------------
    # Related Topics
    # ------------------------

    related = related_finder.find(
        docs,
        topic
    )

    for other in related:

        graph.add_edge(
            topic,
            other
        )

print(graph.number_of_nodes())
print(graph.number_of_edges())

net = Network(
    height="900px",
    width="100%",
    notebook=False
)

net.from_nx(graph)

output_path = os.path.join(
    PROJECT_ROOT,
    "knowledge_graph.html"
)

net.write_html(
    output_path,
    notebook=False
)

print(f"Saved graph to: {output_path}")