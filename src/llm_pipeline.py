import os
import json
from openai import OpenAI
from src.prompts import get_zero_shot_prompt, get_few_shot_prompt
from dotenv import load_dotenv

load_dotenv()

class LLMPipeline:
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found. Defaulting to Mock mode.")
            self.use_mock = True
        
        if not self.use_mock:
            self.client = OpenAI(api_key=self.api_key)

    def _mock_llm(self, prompt: str) -> dict:
        """Fallback mock function if API key is missing."""
        return {
            "intent": "Unknown (Mock)",
            "summary": "This is a mock response because no API key was provided.",
            "response": "Please configure your OPENAI_API_KEY to see real LLM results."
        }

    def query(self, query_text: str, context: list = None, strategy: str = "few-shot") -> dict:
        """Run the LLM pipeline."""
        if strategy == "few-shot" and context:
            context_str = "\n".join([f"- Query: {{c['query']}}\n  Response: {{c['resolution']}}" for c in context])
            prompt = get_few_shot_prompt(query_text, context_str)
        else:
            prompt = get_zero_shot_prompt(query_text)

        if self.use_mock:
            return self._mock_llm(prompt)

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{{"role": "system", "content": "You are a helpful customer support AI."}},
                          {{"role": "user", "content": prompt}}],
                temperature=0
            )
            content = response.choices[0].message.content
            # Basic parsing of JSON from string
            return json.loads(content)
        except Exception as e:
            print(f"LLM Error: {{e}}")
            return self._mock_llm(prompt)
