import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from generation.context_builder import build_context
from generation.prompt import build_prompt


class AnswerGenerator:

    def __init__(self, model_name):

        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()

    def generate(self, question, retrieved_docs):

        # -------------------------------------------------
        # Build Context
        # -------------------------------------------------

        context = build_context(retrieved_docs)

        # -------------------------------------------------
        # Build Prompt
        # -------------------------------------------------

        prompt = build_prompt(
            question=question,
            context=context
        )

        # -------------------------------------------------
        # DEBUG
        # -------------------------------------------------

        print("\n" + "=" * 100)
        print("QUESTION:")
        print(question)
        print("=" * 100)
        print("PROMPT (first 3000 chars):")
        print(prompt[:3000])
        print("=" * 100)

        # -------------------------------------------------
        # Tokenize
        # -------------------------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=3072
        ).to(self.model.device)

        # -------------------------------------------------
        # Free cached memory
        # -------------------------------------------------

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------
        # Generate
        # -------------------------------------------------

        with torch.no_grad():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=250,

                do_sample=False,

                repetition_penalty=1.1,

                use_cache=True,

                eos_token_id=self.tokenizer.eos_token_id,

                pad_token_id=self.tokenizer.eos_token_id

            )

        # -------------------------------------------------
        # Decode ONLY generated text
        # -------------------------------------------------

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        return answer.strip()