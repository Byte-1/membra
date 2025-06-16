import nltk
import asyncio
from typing import List
from .base_chunker import BaseChunker
from nltk.tokenize import PunktSentenceTokenizer
import re

# Download and initialize the tokenizer once
try:
    nltk.download('punkt')
    _tokenizer = PunktSentenceTokenizer()
except Exception:
    nltk.download('punkt', quiet=True)
    _tokenizer = PunktSentenceTokenizer()

class SentenceChunker(BaseChunker):
    def __init__(self, chunk_size: int = 5, overlap: int = 1, debug: bool = False):
        if chunk_size < 1:
            raise ValueError("chunk_size must be at least 1 sentence")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        self.chunk_size = chunk_size
        self.overlap = min(overlap, chunk_size - 1)
        self.debug = debug

    def _normalize_for_tokenizer(self, text: str) -> str:
        text = re.sub(r'\n+', '. ', text)  # convert newlines to periods
        text = re.sub(r'\s+', ' ', text)   # collapse excessive spaces
        return text.strip()

    async def chunk(self, texts: List[str]) -> List[str]:
        if not texts:
            raise ValueError("Input text list is empty. Provide valid content to chunk.")

        chunks = []

        try:
            for idx, text in enumerate(texts):
                if not text.strip():
                    continue

                normalized_text = self._normalize_for_tokenizer(text)
                # Run tokenization in executor since it's CPU-bound
                loop = asyncio.get_event_loop()
                sentences = await loop.run_in_executor(None, _tokenizer.tokenize, normalized_text)

                if self.debug:
                    print(f"\n[SentenceChunker] Text {idx+1}:")
                    print(f"  Total sentences: {len(sentences)}")
                    print(f"  First few sentences: {sentences[:min(5, len(sentences))]}")

                start = 0
                while start < len(sentences):
                    end = start + self.chunk_size
                    chunk_sentences = sentences[start:end]
                    chunk = " ".join(chunk_sentences)
                    chunks.append(chunk)

                    if self.debug:
                        print(f"  Generated chunk ({start}-{end}): {chunk[:120]}... [{len(chunk_sentences)} sentences]")

                    start += self.chunk_size - self.overlap

            return chunks

        except Exception as e:
            raise RuntimeError(f"[SentenceChunker] Failed to chunk due to: {e}") from e