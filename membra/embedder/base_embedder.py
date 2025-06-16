from abc import ABC, abstractmethod
from typing import List 

"""
Takes chunks from your document (text) and converts them into high-dimensional vectors (lists of floats), which are:
    1. Numerical representations of meaning
    2. Used for similarity search, clustering, retrieval, etc.
"""
class BaseEmbedder(ABC):
    @abstractmethod
    async def embed(self, text: List[str]) -> List[float]:
        pass