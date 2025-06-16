from abc import ABC, abstractmethod

class BaseLLMConnector(ABC):
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        pass