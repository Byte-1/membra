from ..base.base_llm_connector import BaseLLMConnector
from ollama import AsyncClient

class OllamaLLMConnector(BaseLLMConnector):
    def __init__(self, model="llama3"):
        self.model = model

    async def stream_generate(self, prompt: str):
        """
        Async generator to stream output from LLM in real-time.
        Ideal for CLI or UI use (WebSockets, SSE).
        """
        message = {
            "role": "user",
            "content": prompt
        }

        client = AsyncClient()
        try:
            async for part in await client.chat(model=self.model, messages=[message], stream=True):
                yield part["message"]["content"]
        finally:
            if hasattr(client, 'close') and callable(client.close):
                await client.close()

    async def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Get complete response from LLM.
        If stream=True, it internally uses streaming but returns the full string.
        """
        if stream:
            return "".join([chunk async for chunk in self.stream_generate(prompt)])
        else:
            message = {
                "role": "user",
                "content": prompt
            }
            client = AsyncClient()
            try:
                response = await client.chat(model=self.model, messages=[message], stream=False)
                return response["message"]["content"]
            finally:
                if hasattr(client, 'close') and callable(client.close):
                    await client.close()
