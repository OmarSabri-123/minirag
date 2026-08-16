from ..LLMInterface import LLMInterface
from ..LLMEnums import VLLMEnums
from logger import logger
from openai import AsyncOpenAI
from typing import Union, List

class VLLMProvider(LLMInterface):

    def __init__(self, api_url: str,
                 api_key: str = "EMPTY",
                 default_input_max_characters: int = 1000,
                 default_output_max_tokens: int = 1000,
                 default_temperature: float = 0.1,
                 enable_thinking: bool = False):

        self.api_url = api_url
        # qwen3 hybrid models emit <think>...</think> blocks by default, which
        # would leak into the rag answer; the flag is ignored by templates
        # that do not define it
        self.enable_thinking = enable_thinking
        # vLLM ignores the key unless the server was started with --api-key,
        # but the OpenAI client refuses to build without a non-empty one
        self.api_key = api_key if api_key and len(api_key) else "EMPTY"

        self.default_input_max_characters = default_input_max_characters
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )

        self.enums = VLLMEnums

        self.logger = logger

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_dimension: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_dimension

    async def generate_text(self, prompt: str, chat_history=[], max_output_tokens: int = None, temperature: float = None):

        if not self.client:
            logger.error("vLLM client is not initialized.")
            return None

        if not self.generation_model_id:
            logger.error("Generation model is not set.")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        chat_history.append(
            self.construt_prompt(prompt, role=VLLMEnums.USER.value)
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.generation_model_id,
                messages=chat_history,
                max_tokens=max_output_tokens,
                temperature=temperature,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": self.enable_thinking}
                }
            )
        except Exception as e:
            logger.error(f"Error while calling vLLM at {self.api_url}: {e}")
            return None

        if not response or not response.choices or response.choices[0].message.content is None:
            logger.error("No response returned from vLLM.")
            return None

        generated_text = response.choices[0].message.content
        return generated_text

    async def embed_text(self, text: Union[str, List[str]], document_type: str = None):

        if not self.client:
            logger.error("vLLM client is not initialized.")
            return None

        if not self.embedding_model_id:
            logger.error("Embedding model is not set.")
            return None

        if isinstance(text, str):
            text = [text]

        try:
            response = await self.client.embeddings.create(
                input=text,
                model=self.embedding_model_id
            )
        except Exception as e:
            # a vLLM server started on a chat model does not expose /v1/embeddings
            logger.error(f"Error while embedding with vLLM at {self.api_url}: {e}")
            return None

        if not response or not response.data or response.data[0].embedding is None:
            logger.error("No embedding returned from vLLM.")
            return None

        vectors = [rec.embedding for rec in response.data]

        # the vector db collection is created with a fixed size, so a mismatch
        # here fails later at insert time with a much less obvious error
        if self.embedding_size and len(vectors[0]) != int(self.embedding_size):
            logger.error(
                f"Embedding dimension mismatch: model '{self.embedding_model_id}' returned "
                f"{len(vectors[0])} but EMBEDDING_MODEL_DIMENSION is {self.embedding_size}."
            )
            return None

        return vectors

    def construt_prompt(self, prompt: str, role: str):

        return {
            "role": role,
            "content": prompt
        }
