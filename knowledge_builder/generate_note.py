import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from knowledge_builder.prompt import build_note_prompt


class NoteGenerator:

    def __init__(self):

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def generate(self, topic, docs):

        # Use only top 10 documents
        docs = docs.head(10)

        prompt = build_note_prompt(
            topic,
            docs
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        with torch.no_grad():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=600,

                do_sample=False,

                repetition_penalty=1.2,

                no_repeat_ngram_size=4,

                pad_token_id=self.tokenizer.eos_token_id,

                eos_token_id=self.tokenizer.eos_token_id

            )

        generated = outputs[0][
            inputs.input_ids.shape[1]:
        ]

        answer = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        )

        return answer.strip()