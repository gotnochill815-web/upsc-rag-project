import google.generativeai as genai


class GeminiGenerator:

    def __init__(self, api_key):

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    def generate(self, prompt):

        response = self.model.generate_content(prompt)

        return response.text