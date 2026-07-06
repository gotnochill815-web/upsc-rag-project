import os
import pandas as pd

PROJECT_ROOT = "/content/drive/MyDrive/upsc_rag_project"

NOTES_DIR = os.path.join(
    PROJECT_ROOT,
    "knowledge_base"
)

rows = []

for file in sorted(os.listdir(NOTES_DIR)):

    if not file.endswith(".md"):
        continue

    topic = file.replace(".md", "")

    with open(
        os.path.join(NOTES_DIR, file),
        "r",
        encoding="utf-8"
    ) as f:

        text = f.read()

    rows.append({

        "doc_id": f"NOTE_{topic.upper().replace(' ','_')}",

        "source": "knowledge_note",

        "title": topic,

        "topic": topic,

        "text": text,

        "retrieval_text": topic + "\n" + text

    })

notes = pd.DataFrame(rows)

print(notes.shape)

notes.to_csv(

    os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "knowledge_notes.csv"
    ),

    index=False

)

print("Saved knowledge_notes.csv")