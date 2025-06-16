from ..base.base_llm_connector import BaseLLMConnector
from openai import OpenAI

class OpenAIConnector(BaseLLMConnector):
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()

