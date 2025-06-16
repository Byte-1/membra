from abc import ABC, abstractmethod
from typing import List

######################## 
# Base Class for Loaders.
# Responsible for loading and parsing documets.
########################

class BaseDocumentLoader(ABC):
    @abstractmethod
    async def load(self, path: str) -> List[str]:
        pass