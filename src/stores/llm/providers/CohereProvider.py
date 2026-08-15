from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
from typing import Union, List
from logger import logger
import cohere

class CohereProvider(LLMInterface):

    def __init__(self, api_key: str,
                 default_input_max_characters: int = 1000,
                 default_output_max_tokens: int = 1000,
                 default_temperature: float = 0.1):
        
        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_output_max_tokens = default_output_max_tokens
        self.default_temperature = default_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        self.enums = CoHereEnums

        self.client = cohere.AsyncClientV2(api_key=self.api_key)

        self.logger = logger
    
    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
    
    def set_embedding_model(self, model_id: str, embedding_dimension: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_dimension

    async def generate_text(self, prompt: str, chat_history=[], max_output_tokens: int = None, temperature: float = None):
        if not self.client:
            logger.error("Cohere client is not initialized.")
            return None

        if not self.generation_model_id:
            logger.error("Generation model is not set.")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens is not None else self.default_output_max_tokens
        temperature = temperature if temperature is not None else self.default_temperature

        chat_history.append(
            self.construt_prompt(prompt, role=CoHereEnums.USER.value)
        )

        response = await self.client.chat(
            model=self.generation_model_id,
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature
        )

        if not response or not response.message.content[0].text:
            logger.error("No response from Cohere API.")
            return None
        
        return response.message.content[0].text
    
    async def embed_text(self, text: Union[str, List[str]], document_type: str = None):

        if not self.client:
            logger.error("Cohere client is not initialized.")
            return None
        
        if not self.embedding_model_id:
            logger.error("Embedding model is not set.")
            return None
        
        input_type = CoHereEnums.DOCUMENT.value
        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = CoHereEnums.QUERY.value

        if isinstance(text, str):
            text = [text]
        
        res = await self.client.embed(
            model = self.embedding_model_id,
            texts = text,
            input_type = input_type,
            embedding_types=["float"],
        )

        if not res or not res.embeddings.float:
            logger.error("No embedding returned from Cohere.")
            return None
        
        
        return [ f for f in res.embeddings.float ]


    def construt_prompt(self, prompt: str, role: str):

        return {
            "role": role,
            "content": prompt
        }
