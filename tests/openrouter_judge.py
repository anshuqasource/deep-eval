import os
from dotenv import load_dotenv
load_dotenv()

from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI

class OpenRouterJudge(DeepEvalBaseLLM):
    def __init__(self, model="meta-llama/llama-3.3-70b-instruct"):
        self.model_name = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name