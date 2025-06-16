from membra.llmconnector.base.base_llm_connector import BaseLLMConnector
import aiohttp

class PhiLLMConnector(BaseLLMConnector):
    def __init__(self, model: str = "phi", max_tokens: int = 200, temperature: float = 0.7):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_url = "http://localhost:11434/api/generate"

    async def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
                "stop": ["\nUser:", "\nQuestion:"]
            }
        }

        try:
            print(f"Choosen model is {self.model}")
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["response"]
        except aiohttp.ClientError as e:
            raise RuntimeError(f"[PhiLLMConnector] Request failed: {e}")
        except KeyError:
            raise ValueError("[PhiLLMConnector] Unexpected response structure")
