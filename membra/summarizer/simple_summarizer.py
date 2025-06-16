import time
from rich import print
from membra.summarizer.base_summarizer import BaseSummarizer
import asyncio

class Summarizer(BaseSummarizer):
    def __init__(self, llm, augmenter):
        super().__init__(llm=llm, augmenter=augmenter)

    async def _summarize_chunk(self, chunk: str) -> str:
        start_time = time.perf_counter()
        prompt = self.augmenter.build_summary_prompt(chunk)

        if prompt is None:
            return chunk.strip()

        try:
            summary = await self.llm.generate(prompt)
            print(f"[bold blue] Chunk[/bold blue]: {chunk}\n[bold blue] Summary[/bold blue]: {summary.strip()}")
            return summary.strip()
        except Exception as e:
            print(f"[bold red][Summarizer] Failed to summarize chunk: {e}[/bold red]")
            return "[Error summarizing this chunk]"
        finally:
            end_time = time.perf_counter()
            print(f"[Summary] Took {end_time - start_time:.2f} seconds")

    async def summarize_chunks(self, chunks):
        tasks = [self._summarize_chunk(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)
