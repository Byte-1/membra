from abc import ABC, abstractmethod
from ..vectorstore.base_vectorstore import BaseVectorStore
from typing import List

class BaseRetriever(ABC):
    def __init__(self, vectorstore: BaseVectorStore):
        self.vectorstore = vectorstore

    @abstractmethod
    async def retrieve(self, query_text: str, project_id: str, top_k: int = 5, min_score: float = 0.0) -> List[str]:
        """
        Given a query and a project, return top_k most relevant document chunks.

        Args:
            query_text (str): The user's query.
            project (str): Identifier for the document collection.
            top_k (int): Number of top results to return.
            min_score (float): Minimum cosine similarity threshold.

        Returns:
            List[str]: Ranked list of relevant chunks.
        """
        pass
