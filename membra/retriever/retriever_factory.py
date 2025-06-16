from .base_retriever import BaseRetriever
from .simple_retriever import SimpleRetriever
from ..vectorstore.base_vectorstore import BaseVectorStore

class RetrieverFactory:
    __retriever_registry = {
        "simple": SimpleRetriever
    }

    def get_retriever(self, retriever_type: str, vectorstore: BaseVectorStore) -> BaseRetriever:
        if retriever_type not in self.__retriever_registry:
            raise KeyError(f"No Retriever of type: {retriever_type}")
        
        retriever_cls = self.__retriever_registry[retriever_type]
        return retriever_cls(vectorstore)
