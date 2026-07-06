import os
import pandas as pd

from collect_documents import DocumentCollector
from keyword_extractor import KeywordExtractor
from theme_extractor import ThemeExtractor
from related_topics import RelatedTopicFinder
# =====================================================
# Project Paths
# =====================================================

PROJECT_ROOT = "/content/drive/MyDrive/upsc_rag_project"

OUTPUT = os.path.join(
    PROJECT_ROOT,
    "knowledge_base_structured"
)

os.makedirs(
    OUTPUT,
    exist_ok=True
)

# =====================================================
# Initialize
# =====================================================

collector = DocumentCollector()

keyword_extractor = KeywordExtractor()

theme_extractor = ThemeExtractor()

related_finder = RelatedTopicFinder(
    collector.config
)

# =====================================================
# Build One Topic
# =====================================================

def build_topic(topic):

    docs = collector.collect(topic)

    constitution = (
        docs[
            docs["source"] == "constitution"
        ]
        .sort_values("doc_id")
    )

    pyqs = (
        docs[
            docs["source"] == "pyq"
        ]
        .sort_values("title")
    )

    keywords = keyword_extractor.extract(docs)

    themes = theme_extractor.extract(docs)
    related_topics = related_finder.find(
       docs,
       topic
   )

    note = []

    # =====================================================
    # Title
    # =====================================================

    note.append(f"# {topic}\n")

    # =====================================================
    # Overview
    # =====================================================

    note.append("## Overview\n")

    note.append(
        f"This note consolidates Constitutional provisions, "
        f"relevant UPSC Previous Year Questions (PYQs), "
        f"recurring themes, and important revision keywords "
        f"related to **{topic}**.\n"
    )

    # =====================================================
    # Constitutional Articles
    # =====================================================

    note.append("## Constitutional Articles\n")

    if constitution.empty:

        note.append("No Constitutional Articles found.\n")

    else:

        for row in constitution.itertuples():

            note.append(f"### {row.doc_id}")

            note.append(f"**{row.title}**\n")

            text = "" if pd.isna(row.text) else str(row.text)

            note.append(text)

            note.append("\n---\n")

    # =====================================================
    # Previous Year Questions
    # =====================================================

    note.append("## Previous Year Questions\n")

    if pyqs.empty:

        note.append("No Previous Year Questions found.\n")

    else:

        for row in pyqs.itertuples():

            note.append(f"### {row.title}")

            text = "" if pd.isna(row.text) else str(row.text)

            note.append(text)

            note.append("")

        # =====================================================
    # Frequently Asked Themes
    # =====================================================

    note.append("## Frequently Asked Themes\n")

    if len(themes) == 0:

        note.append("No recurring themes identified.\n")

    else:

        for theme in themes:

            note.append(f"- {theme}")

    note.append("")

    # =====================================================
    # Related Topics
    # =====================================================

    note.append("## Related Topics\n")

    if len(related_topics) == 0:

        note.append("No directly related topics found.\n")

    else:

        for item in related_topics:

            note.append(f"- {item}")

    note.append("")

    # =====================================================
    # Important Keywords
    # =====================================================

    note.append("## Important Keywords\n")

    if len(keywords) == 0:

        note.append("No keywords extracted.\n")

    else:

        for keyword in keywords:

            note.append(f"- {keyword}")

    note.append("")

    return "\n".join(note)