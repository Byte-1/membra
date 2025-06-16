import faiss
import os
import asyncio
from typing import List,Dict
from membra.vectorstore.base_vectorstore import BaseVectorStore
from membra.embedder.base_embedder import BaseEmbedder
import numpy as np
import pickle

class FaissVectorStore(BaseVectorStore):
    def __init__(self, embedder:BaseEmbedder, index_dir:str = "./vector_idx_dir"):
        super().__init__(embedder)
        self.index_dir = index_dir
        self.project_indexes = {}       # project → FAISS index
        self.project_id_map = {}        # project → {int_id: {"text": str, "metadata": dict}}
        self.project_counter = {}       # project → int counter for new chunk ids
        
    async def initialize(self):
        """Initialize the vector store by loading existing indices and setting up the embedder dimension"""
        hello_embed = await self.embedder.embed(["hello"])
        self.embedder_dimension = len(hello_embed[0])  # To get dimensions that the current embedder uses
        await self.load()

    @classmethod
    async def create(cls, embedder:BaseEmbedder, index_dir:str = "./vector_idx_dir"):
        """Async factory method to create and initialize a FaissVectorStore"""
        instance = cls(embedder, index_dir)
        await instance.initialize()
        return instance

    def _init_project(self, project:str):
        ### Initialize a New project in case not present currently
        index = faiss.IndexIDMap(faiss.IndexFlatIP(self.embedder_dimension))
        self.project_indexes[project] = index
        self.project_id_map[project] = {}
        self.project_counter[project] = 0
    
    async def add(self, project:str, chunks:List[str]):
        if not chunks:
            return
        if project not in self.project_indexes:
            self._init_project(project)
        
        index = self.project_indexes[project]
        id_map = self.project_id_map[project]
        counter = self.project_counter[project]

        embeddings = await self.embedder.embed(chunks)
        embeddings = np.array([self._normalize(e) for e in embeddings]).astype("float32")
        
        ids = np.arange(counter, counter + len(chunks))
        index.add_with_ids(embeddings, ids)

        for i, chunk in zip(ids, chunks):
            id_map[i] = {"text": chunk, "metadata": {}}

        self.project_counter[project] += len(chunks)
        await self.persist()
    
    async def query(self, project: str, query_text: str, top_k: int = 5, min_score: float = 0.0) -> List[Dict]:
        if project not in self.project_indexes:
            raise ValueError(f"[Faiss VectorStore] Project {project} not found in store")
        
        query_embedding = self._normalize((await self.embedder.embed([query_text]))[0]).reshape(1, -1).astype("float32")
        index = self.project_indexes[project]
        id_map = self.project_id_map[project]

        # Search method is exposed by faiss itself for its index
        scores, ids = index.search(query_embedding, top_k)
        results = []
        for i, score in zip(ids[0], scores[0]):
            if i == -1 or score < min_score:
                continue
            item = id_map.get(i)
            if item:
                results.append({
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "score": float(score)
                })
        return results
    
    async def delete_project(self, project: str):
        if project in self.project_indexes:
            del self.project_indexes[project]
            del self.project_id_map[project]
            del self.project_counter[project]

            index_path = os.path.join(self.index_dir, f"{project}.index")
            map_path = os.path.join(self.index_dir, f"{project}_map.pkl")

            if os.path.exist(index_path): os.remove(index_path)
            if os.path.exist(map_path): os.remove(map_path)

    def list_projects(self) -> List[str]:
        return list(self.project_indexes.keys())

    async def persist(self):
        loop = asyncio.get_event_loop()
        
        def _save_index():
            os.makedirs(self.index_dir, exist_ok=True)
            for project, index in self.project_indexes.items():
                faiss.write_index(index, os.path.join(self.index_dir, f"{project}.index"))
                with open(os.path.join(self.index_dir, f"{project}_map.pkl"), "wb") as f:
                    pickle.dump(self.project_id_map[project], f)
        
        await loop.run_in_executor(None, _save_index)

    async def load(self):
        if not os.path.exists(self.index_dir):
            return

        loop = asyncio.get_event_loop()
        
        def _load_indices():
            loaded_data = {}
            for file in os.listdir(self.index_dir):
                if file.endswith(".index"):
                    project = file[:-6]
                    index_path = os.path.join(self.index_dir, file)
                    map_path = os.path.join(self.index_dir, f"{project}_map.pkl")

                    index = faiss.read_index(index_path)
                    with open(map_path, "rb") as f:
                        id_map = pickle.load(f)

                    loaded_data[project] = (index, id_map)
            return loaded_data

        loaded_data = await loop.run_in_executor(None, _load_indices)
        
        for project, (index, id_map) in loaded_data.items():
            max_id = max(id_map.keys(), default=-1) + 1
            self.project_indexes[project] = index
            self.project_id_map[project] = id_map
            self.project_counter[project] = max_id
    
    def _normalize(self, vec: List[float]) -> np.ndarray:
        vec = np.array(vec, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec