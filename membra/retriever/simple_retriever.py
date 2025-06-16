from .base_retriever import BaseRetriever
from ..vectorstore.base_vectorstore import BaseVectorStore
from typing import List
from typing import List, Dict,Union
import re
import unicodedata

class SimpleRetriever(BaseRetriever):
    def __init__(self,vectorstore: BaseVectorStore):
        super().__init__(vectorstore)
    
    def normalize(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)  # Normalize unicode
        text = re.sub(r'\s+', '', text)             # Remove all whitespace (spaces, newlines, tabs)
        text = re.sub(r'[^\w]', '', text)           # Remove punctuation
        text = text.lower().strip()                 # Lowercase
        return text
    
    async def retrieve(self, query_text, project_id, top_k = 5, min_score = 0.0) -> List[str]:
        if not query_text:
            raise ValueError("[SimpleRetriever] query_text cannot be empty.")
        if not project_id:
            raise ValueError("[SimpleRetriever] project name cannot be empty.")
        raw_chunks = await self.vectorstore.query(project=project_id, query_text=query_text, top_k=top_k, min_score=min_score)
        clean_chunks = self.deduplicate_chunks(raw_chunks)
        return clean_chunks
    
    def deduplicate_chunks(self, chunks: List[Union[str, Dict]]) -> List[str]:
        seen = set()
        deduped = []
        chunks = [
            chunk["text"] if isinstance(chunk, dict) and "text" in chunk else chunk
            for chunk in chunks
        ]
        for i, text in enumerate(chunks):
            key = self.normalize(text)
            if key not in seen:
                seen.add(key)
                deduped.append(text)

        return deduped