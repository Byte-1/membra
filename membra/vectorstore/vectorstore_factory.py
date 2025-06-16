from .base_vectorstore import BaseVectorStore
from .faiss_vectorstore import FaissVectorStore
from .in_memory_vectorstore import InMemoryVectorStore


class VectorStoreFactory:
    __vectorstore_registry = {
        "faiss": FaissVectorStore,
        "default": InMemoryVectorStore
    }

    @staticmethod
    async def get_vectorstore(store_name: str, *args) -> BaseVectorStore:
        if store_name not in VectorStoreFactory.__vectorstore_registry:
            raise ValueError(f"Vectorstore {store_name} not supported")

        vectorstore_class = VectorStoreFactory.__vectorstore_registry[store_name]
        vectorstore = vectorstore_class(*args)
        await vectorstore.initialize()
        return vectorstore