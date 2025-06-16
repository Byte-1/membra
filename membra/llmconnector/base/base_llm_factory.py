from abc import ABC, abstractmethod
from membra.llmconnector.base.base_llm_connector import BaseLLMConnector

class BaseLLMFactory(ABC):
    @abstractmethod
    def get_llm(mode: str, **kwargs) -> BaseLLMConnector:
        pass