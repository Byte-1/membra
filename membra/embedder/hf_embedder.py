from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbedder
from typing import List

class HFEmbedder(BaseEmbedder):
    def __init__(self ,model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model.encode(["warmup"])
    
    async def embed(self, text: List[str]) -> List[List[float]]:
        if not text:
            raise ValueError(" [HFEmbedder] Input list is empty. Provide valid content to embed.")
        if not isinstance(text,list):
            raise TypeError("[HFEmbedder] Input to embed() must be a list of strings.")
        
        for t in text:
            if not isinstance(t,str):
                raise TypeError("[HFEmbedder] Every element in the input list to embed() must be a string.")
        try:
            # Run the potentially blocking encode operation in a thread pool
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.encode(text, convert_to_tensor=False).tolist()
            )
            return result
        except Exception as e:
            raise RuntimeError(f"[HFEmbedder] Failed to generate embeddings due to: {e}") from e
