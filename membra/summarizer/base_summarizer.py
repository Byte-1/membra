from abc import ABC, abstractmethod
from membra.llmconnector.base.base_llm_connector import BaseLLMConnector
from membra.prompt.prompt_augmenter import PromptAugmenter
from typing import List

class BaseSummarizer(ABC):
    __default_workers = 10  # "private" class variable

    def __init__(self, llm: BaseLLMConnector , augmenter: PromptAugmenter):
        self.llm = llm
        self.augmenter = augmenter

    @abstractmethod
    async def _summarize_chunk(self, chunk:str) -> str:
        pass

    @abstractmethod
    async def summarize_chunks(self,chunks: List[str]) -> List[str]:
        pass