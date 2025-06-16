from .base_chunker import BaseChunker
from .simple_chunker import SimpleChunker
from .sentence_chunker import SentenceChunker
from .token_chunker import TokenChunker

class ChunkerFactory:
    __chunker_registry = {
        "simple": SimpleChunker,
        "token": TokenChunker,
        "sentence": SentenceChunker
    }
    def get_chunker(self, method: str = "simple", **kwargs) -> BaseChunker:
        if method not in self.__chunker_registry:
            raise ValueError(f"Unknown chunker method: {method}")
        return self.__chunker_registry[method](**kwargs)
