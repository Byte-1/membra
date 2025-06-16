from membra.llmconnector.base.base_llm_connector import BaseLLMConnector
from membra.llmconnector.base.base_llm_factory import BaseLLMFactory
from membra.llmconnector.online_llm.openai_llm_connector import OpenAIConnector

class OnlineLLMFactory(BaseLLMFactory):
    __connector_registry = {
        "openai" : OpenAIConnector 
    }

    @staticmethod
    def get_llm(mode: str, **kwargs) -> BaseLLMConnector:
        if mode not in __class__.__connector_registry:
            raise ValueError(f"Unsupported LLM mode: {mode}")
        return __class__.__connector_registry[mode](**kwargs)
