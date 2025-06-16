from abc import ABC, abstractmethod
from ..embedder.base_embedder import BaseEmbedder
from typing import List,Dict

class BaseVectorStore(ABC):
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder
        
    async def initialize(self):
        """Initialize the vector store. Override this method for async initialization if needed."""
        pass

    @abstractmethod
    async def add(self, project: str, chunks: List[str]):
        """Add vectors and metadata to a specific project."""
        pass

    @abstractmethod
    async def query(self, project:str, query_text: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        """Search within a specific project"""
        pass

    @abstractmethod
    async def delete_project(self, project:str):
        """Delete a project from the vector store."""
        pass

    @abstractmethod
    def list_projects(self) -> List[str]:
        """List all active projects in the vector store."""
        pass

    @abstractmethod
    def persist(self):
        """Persist data to disk or external storage."""
        pass

    @abstractmethod
    def load(self):
        """Load data to disk if present from previous sessions to supprt persistance"""