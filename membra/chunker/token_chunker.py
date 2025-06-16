from typing import List
from .base_chunker import BaseChunker

class TokenChunker(BaseChunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if chunk_size < 32:
            raise ValueError("chunk_size must be at least 32 tokens")
        
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        
        if overlap >= chunk_size:
            self.overlap = chunk_size // 4  # default to 25% if invalid
        else:
            self.overlap = overlap

        self.chunk_size = chunk_size

    def _simple_tokenizer(self, text: str) -> List[str]:
        return text.split()  # Replace with tiktoken in production

    def chunk(self, texts: List[str]) -> List[str]:
        if not texts:
            raise ValueError("Input text list is empty. Provide valid content to chunk.")

        chunks = []
        try: 
            for text in texts:
                if not isinstance(text, str):
                    raise TypeError(f"Expected text to be string but got {type(text)}")
                tokens = self._simple_tokenizer(text)
                start = 0
                while start < len(tokens):
                    end = min(start + self.chunk_size, len(tokens))
                    chunk = " ".join(tokens[start:end])
                    chunks.append(chunk)
                    start += self.chunk_size - self.overlap
            return chunks
        except Exception as e:
            raise RuntimeError(f"[TokenChunker] Failed to chunk due to: {e}") from e
