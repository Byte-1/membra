from .base_chunker import BaseChunker
from typing import List

"""
    Overlap is used in order to make sure that the context is captured.
    i.e. The next chunk will have some data from previous chunk.
    Because: 
        1. Context matters for LLMs.
        2. If chunks are too “cleanly cut,” the model might miss context between them (e.g., incomplete sentences, missing references).
        3. Overlap provides continuity, like a “sliding window.”
    For ex: ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"], chunk_size = 5, overlap = 2
        Chunk 1: The quick brown fox jumps
        Chunk 2:         fox jumps over the lazy
        Chunk 3:                        the lazy dog
"""

class SimpleChunker(BaseChunker):
    def __init__(self, chunk_size: int = 500, overlap: int = None):
        if chunk_size < 20:
            raise ValueError("chunk_size must be at least 50")

        self.chunk_size = chunk_size

        # Auto-adjust overlap if not given or invalid
        if overlap is None:
            self.overlap = int(0.1 * chunk_size)  # default 10%
        elif overlap >= chunk_size or overlap < 0:
            self.overlap = int(0.2 * chunk_size)  # cap at 20%
        else:
            self.overlap = overlap
    
    def chunk(self, texts: List[str]) -> List[str]:
        if not texts:
            raise ValueError("Input text list is empty. Provide valid content to chunk.")
        chunks = []
        try:
            for text in texts:
                words = text.split()
                start = 0
                while start < len(words):
                    end = min(start + self.chunk_size,len(words))
                    cur_chunk = " ".join(words[start:end])
                    chunks.append(cur_chunk)
                    start += self.chunk_size - self.overlap
            return chunks
        except Exception as e:
            raise RuntimeError(f"[SimpleChunker] Failed to chunk due to: {e}") from e



