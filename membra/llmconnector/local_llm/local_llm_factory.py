from membra.llmconnector.base.base_llm_connector import BaseLLMConnector
from membra.llmconnector.base.base_llm_factory import BaseLLMFactory
from membra.llmconnector.local_llm.ollama_local_llm_connector import OllamaLLMConnector
from membra.llmconnector.local_llm.phi_llm_connector import PhiLLMConnector

class LocalLLMFactory(BaseLLMFactory):
    __connector_registry = {
        "llama3" : OllamaLLMConnector,
        "phi": PhiLLMConnector 
    }

    @staticmethod
    def get_llm(mode: str, **kwargs) -> BaseLLMConnector:
        if mode not in __class__.__connector_registry:
            raise ValueError(f"Unsupported LLM mode: {mode}")
        return __class__.__connector_registry[mode](**kwargs)
