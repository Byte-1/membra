from typing import Dict, List, Tuple
import numpy as np
import heapq
import os
import pickle
from .base_vectorstore import BaseVectorStore
from ..utils.similarity_check import cosine_similarity

class InMemoryVectorStore(BaseVectorStore):
    def __init__(self, embedder, storage_path: str = "vectorstore.pkl"):
        super().__init__(embedder)
        self.store: Dict[str, List[Tuple[str, List[float]]]] = {} # self.store: {project: [(chunk1, [vectors]),(chunk2,[vectors])]}
        self.storage_path = storage_path

    def add(self, project: str, chunks: List[str]):
        if not chunks:
            raise ValueError("[VectorStore:add] Cannot add empty chunk list.")
        if not isinstance(chunks, list):
            raise TypeError("[VectorStore:add] Chunks must be a list of strings.")
        if any(not isinstance(chunk, str) for chunk in chunks):
            raise TypeError("[VectorStore:add] All chunks must be strings.")

        try:
            vectors = self.embedder.embed(chunks)
            self.store[project] = list(zip(chunks, vectors))
        except Exception as e:
            raise RuntimeError(f"[VectorStore:add] Failed to add chunks due to: {e}") from e

    def query(self, project: str, query_text: str, top_k: int = 5, min_score: float = 0.0) -> List[str]:
        if project not in self.store:
            raise ValueError(f"[VectorStore:query] Project '{project}' not found in store.")
        if not query_text:
            raise ValueError("[VectorStore:query] Query text is empty.")
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("[VectorStore:query] top_k must be a positive integer.")
        if min_score < 0.0 or min_score > 1.0:
            raise ValueError(f"[VectorStore:query] min_score should be between 0.0 and 1.0")

        try:
            query_vec = self.embedder.embed([query_text])[0] ## Embed returns a list of list of floats that's why we have taken [0]
            results = []

            for chunk, vec in self.store[project]:
                score = cosine_similarity(query_vec, vec)
                if score > min_score:
                    results.append((score, chunk))

            top_chunks = heapq.nlargest(top_k, results)
            return [chunk for _, chunk in top_chunks]
        except Exception as e:
            raise RuntimeError(f"[VectorStore:query] Query failed due to: {e}") from e

    def delete_project(self, project: str):
        if project not in self.store:
            raise ValueError(f"[VectorStore:delete] Project '{project}' does not exist.")
        del self.store[project]

    def list_projects(self) -> List[str]:
        return list(self.store.keys())


    def persist(self) -> None:
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'rb') as f:
                    existing_data = pickle.load(f)
                existing_data.update(self.store)
                self.store = existing_data
                
            with open(self.storage_path, 'wb') as f:
                pickle.dump(self.store, f)
        except Exception as e:
            raise RuntimeError(f"[VectorStore:persist] Failed to persist store: {e}")

    def load(self) -> None:
        if not os.path.exists(self.storage_path):
            return  # no-op if file doesn't exist
        try:
            with open(self.storage_path, 'rb') as f:
                self.store = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"[VectorStore:load] Failed to load store: {e}")
