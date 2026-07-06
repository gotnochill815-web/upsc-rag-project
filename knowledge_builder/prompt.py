def build_note_prompt(topic, documents):

    context = ""

    for i, row in enumerate(documents.itertuples(index=False), start=1):

        text = str(row.text)

        if text == "nan":
            text = ""

        # Limit each document
        text = text[:700]

        context += f"""
==============================
Document {i}
==============================

Title:
{row.title}

Source:
{row.source}

Content:
{text}

"""

    prompt = f"""
You are an expert UPSC Civil Services GS-II faculty.

Your task is to prepare concise UPSC revision notes.

IMPORTANT RULES

1. Use ONLY the retrieved documents.

2. Do NOT use outside knowledge.

3. Do NOT invent Constitutional Articles.

4. Do NOT invent PYQs.

5. Do NOT continue the document numbering.

6. Do NOT generate additional documents.

7. If information is unavailable,
simply omit that point.

8. Merge duplicate information.

9. Keep notes concise.

==============================
RETRIEVED DOCUMENTS
==============================

{context}

==============================
END OF RETRIEVED DOCUMENTS
==============================

Now prepare revision notes.

Structure exactly as follows:

# {topic}

## Introduction

## Constitutional Provisions

## Key Features

## Important Articles

## Previous Year Question Themes

## Key Takeaways

Notes:
"""

    return prompt