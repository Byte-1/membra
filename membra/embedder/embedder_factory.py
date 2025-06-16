from .base_embedder import BaseEmbedder
from .hf_embedder import HFEmbedder
from .openai_embedder import OpenAIEmbedder

class Embedderfactory:
    __embedder_registry = {
        "hf" : HFEmbedder,
        "openai": OpenAIEmbedder
    }

    def get_embedder(self, method="hf", **kwargs) -> BaseEmbedder:
        if method not in self.__embedder_registry:
            raise ValueError(f"Unsupported embedder: {method}")
        return self.__embedder_registry[method](**kwargs)