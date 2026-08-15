from .LLMEnums import LLMEnums
from .providers import OpenAIProvider, CohereProvider, VLLMProvider

class LLMProviderFactory:

    def __init__(self, config: dict):
        self.config = config
    
    def create(self, provider: str, for_embedding: bool = False):

        if provider == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=self.config.OPENAI_API_URL,
                default_input_max_characters=self.config.DAFAULT_INPUT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.DAFAULT_OUTPUT_MAX_TOKENS,
                default_temperature=self.config.DAFAULT_TEMPERATURE
            )
        
        if provider == LLMEnums.COHERE.value:
            return CohereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_characters=self.config.DAFAULT_INPUT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.DAFAULT_OUTPUT_MAX_TOKENS,
                default_temperature=self.config.DAFAULT_TEMPERATURE
            )

        if provider == LLMEnums.VLLM.value:
            # generation and embedding are served by two separate vLLM containers,
            # so each needs its own base url
            api_url = self.config.VLLM_API_URL
            if for_embedding and self.config.VLLM_EMBEDDING_API_URL:
                api_url = self.config.VLLM_EMBEDDING_API_URL

            return VLLMProvider(
                api_url=api_url,
                api_key=self.config.VLLM_API_KEY,
                default_input_max_characters=self.config.DAFAULT_INPUT_MAX_CHARACTERS,
                default_output_max_tokens=self.config.DAFAULT_OUTPUT_MAX_TOKENS,
                default_temperature=self.config.DAFAULT_TEMPERATURE
            )

        return None