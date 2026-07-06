import os
import sys

# ---------------------------------------------------
# Project Root
# ---------------------------------------------------

try:
    PROJECT_ROOT = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
except NameError:
    PROJECT_ROOT = "/content/drive/MyDrive/upsc_rag_project"

sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------
# Imports
# ---------------------------------------------------

from knowledge_builder.collect_documents import DocumentCollector
from knowledge_builder.generate_note import NoteGenerator

# ---------------------------------------------------
# Output Folder
# ---------------------------------------------------

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "knowledge_base"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ---------------------------------------------------
# Initialize
# ---------------------------------------------------

collector = DocumentCollector()

generator = NoteGenerator()

# ---------------------------------------------------
# Generate Notes
# ---------------------------------------------------

for topic in collector.config["topics"]:

    print("=" * 70)
    print(f"Generating notes for: {topic}")

    docs = collector.rank_documents(
        topic,
        top_k=10
)

    print(f"Retrieved {len(docs)} documents")

    note = generator.generate(
        topic=topic,
        docs=docs
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        f"{topic}.md"
    )

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(note)

    print(f"Saved -> {save_path}")

print("\n All notes generated successfully!")