from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"

class Settings(BaseSettings):

    APP_NAME: str

    FILE_ALLOWED_EXTENSIONS: list
    MAX_FILE_SIZE: int  
    FILE_DEFAULT_CHUNK_SIZE: int  # in bytes

    POSTGRES_USERNAME: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str
    POSTGRES_PORT: str
    POSTGRES_MAIN_DATABASE: str

    DAFAULT_INPUT_MAX_CHARACTERS: Optional[int] = None
    DAFAULT_OUTPUT_MAX_TOKENS: Optional[int] = None
    DAFAULT_TEMPERATURE: Optional[float] = None
    
    GENERATION_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_ID: Optional[str] = None
    EMBEDDING_MODEL_DIMENSION: Optional[str] = None

    COHERE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_API_URL: Optional[str] = None

    VLLM_API_URL: Optional[str] = None
    VLLM_EMBEDDING_API_URL: Optional[str] = None
    VLLM_API_KEY: Optional[str] = "EMPTY"

    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str

    VECTOR_DB_BACKEND: str
    QDRANT_URL: Optional[str] = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_PREFER_GRPC: bool = False
    VECTOR_DB_DISTANCE_METHOD: Optional[str] = None
    INDEX_THRESHOLD: int

    RERANK_CROSS_ENCODER_NAME: Optional[str] = None

    CHUNK_TOKENIZER_NAME: str = "xlm-roberta-base"
    CHUNK_SIZE_TOKENS: int = 400
    CHUNK_OVERLAP_TOKENS: int = 80

    PRIMARY_LANG: Optional[str] = None
    DEFAULT_LANG: str

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        extra="ignore",
    )


def get_settings():
    return Settings()