class PromptBuilder:

    def build(self, question, contexts):

        context_text = ""

        for i, row in contexts.iterrows():

            context_text += f"""
Document {i+1}

Title:
{row["title"]}

Content:
{row["text"]}

---------------------------------------
"""

        prompt = f"""
You are an expert UPSC GS-II answer writing assistant.

Use ONLY the supplied documents.

Question:
{question}

Reference Documents

{context_text}

Instructions

Write a UPSC Mains answer.

Structure:

Introduction

Body

Constitutional Articles

Committee / Commission (if applicable)

Examples

Way Forward

Conclusion

Word limit:
250 words.

Do not hallucinate.
"""

        return prompt