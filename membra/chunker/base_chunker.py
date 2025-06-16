from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    @abstractmethod
    async def chunk(self, texts: List[str]) -> List[str]:
        """
        Takes a list of page texts and returns a list of clean chunks.
        """
        pass