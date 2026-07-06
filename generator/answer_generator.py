import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class AnswerGenerator:

    def __init__(self):

        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,   # CPU deployment
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )

        self.model.eval()

    def build_prompt(self, question, docs):

        context = ""

        for i, row in enumerate(docs.itertuples(), 1):

            context += f"""
==============================
Document {i}

Title: {row.title}

Source: {row.source}

Content:
{row.text}
==============================
"""

        return f"""
You are an expert UPSC GS-II mentor.

You MUST answer ONLY using the retrieved documents below.

Rules:
1. Do NOT use outside knowledge.
2. Do NOT invent constitutional articles.
3. Do NOT invent facts.
4. If the retrieved documents do not contain sufficient information, explicitly write:
   "Information not found in retrieved documents."
5. Use only evidence present in the retrieved context.

Write the answer in UPSC Mains format:

# Introduction

# Constitutional Provisions

# Analysis

# Way Forward

# Conclusion

Question:
{question}

Retrieved Documents:

{context}
"""

    def generate(self, question, docs):

        prompt = self.build_prompt(question, docs)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert UPSC GS-II faculty. "
                    "Answer strictly from the provided documents."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        )

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=600,
                temperature=0.1,
                top_p=0.9,
                do_sample=False,
                repetition_penalty=1.15,
                pad_token_id=self.tokenizer.eos_token_id
            )

        answer = outputs[0][inputs.input_ids.shape[1]:]

        return self.tokenizer.decode(
            answer,
            skip_special_tokens=True
        ).strip()
