from membra.llmconnector.base.base_llm_factory import BaseLLMFactory
from membra.llmconnector.local_llm.local_llm_factory import LocalLLMFactory
from membra.llmconnector.online_llm.online_llm_factory import OnlineLLMFactory

class LLMFactory:
    __factory_registry = {
        "local" : LocalLLMFactory,
        "online" : OnlineLLMFactory 
    }

    @staticmethod
    def get_llm_factory(mode: str, **kwargs) -> BaseLLMFactory:
        if mode not in __class__.__factory_registry:
            raise ValueError(f"Unsupported LLM mode: {mode}")
        return __class__.__factory_registry[mode](**kwargs)
