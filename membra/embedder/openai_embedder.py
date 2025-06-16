import openai
from .base_embedder import BaseEmbedder
from typing import List

class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, model_name="text-embedding-3-small", api_key=None):
        if not api_key:
            raise ValueError("OpenAI API key is required for OpenAIEmbedder.")
        self.model = model_name
        openai.api_key = api_key

    def embed(self, text: List[str]) -> List[List[float]]:
        if not text:
            raise ValueError("Input list is empty. Provide valid content to embed.")
        if not isinstance(text, list):
            raise TypeError("Input to embed() must be a list of strings.")
        
        for t in text:
            if not isinstance(t,str):
                raise TypeError("[OpenAIEmbedder] Every element in the input list to embed() must be a string.")
        try:
            response = openai.embeddings.create(model=self.model, input=text)
            return [item.embedding for item in response.data]
        except Exception as e:
            raise RuntimeError(f"[OpenAIEmbedder] Failed to generate embeddings due to: {e}") from e
