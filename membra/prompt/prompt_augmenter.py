from typing import List,Union
import math 

class PromptAugmenter:
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or (
            "You are an intelligent assistant. Use the following context to answer the user's question as accurately as possible."
        )

    def build_prompt(self, query: str, context_chunks: List[Union[str, dict]]) -> str:
        # Normalize context_chunks to list of strings
        processed_chunks = [
            chunk["text"] if isinstance(chunk, dict) and "text" in chunk else chunk
            for chunk in context_chunks
        ]
        context_text = "\n\n".join(processed_chunks)
        return f"{self.system_prompt}\n\nContext:\n{context_text}\n\nQuestion:\n{query}\n\nAnswer:"
    
    def build_summary_prompt(self, chunk: str) -> str | None:
        """Generate a prompt to summarize a chunk based on word count.
        
        Returns None if the chunk is too short to summarize.
        """
        stripped = chunk.strip()
        words = stripped.split()
        word_count = len(words)

        if word_count <= 30:
            # Too short to summarize meaningfully — skip LLM call
            return None

        n = max(5, math.ceil(word_count * 0.2))  # 20% of original length, at least 5 words

        return (
            f"Compress the following content to approximately {n} words (around 20% of the original), "
            f"while preserving all key concepts, context, and important details. "
            f"Do not include any greetings or explanations. "
            f"Only return the summary. Strictly stay within the word limit.\n\n"
            f"{stripped}"
        )
