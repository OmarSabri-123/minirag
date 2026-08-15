from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.base import base_router
from routes.upload import upload_router
from routes.nlp import nlp_router
from helpers import get_settings, Settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from utils.metrics import setup_metrics
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sentence_transformers import CrossEncoder
import torch

app = FastAPI(title="Multi-Model RAG API")

setup_metrics(app)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_credentials=["*"],
#     allow_headers=["*"],
# )

# git rm -r --cached assets/database/
async def startup_span():

    settings: Settings = get_settings()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device == 'cuda' else torch.float32

    postgres_conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"

    app.db_engine = create_async_engine(postgres_conn)
    
    app.db_client = sessionmaker(
        bind=app.db_engine, class_=AsyncSession, expire_on_commit=False
    )

    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(settings, db_client=app.db_client)

    app.generation_client = llm_provider_factory.create(
        settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(settings.GENERATION_MODEL_ID)

    app.embedding_client = llm_provider_factory.create(
        settings.EMBEDDING_BACKEND, for_embedding=True)
    app.embedding_client.set_embedding_model(
        settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_MODEL_DIMENSION)

    app.vectordb_client = vectordb_provider_factory.create(
        settings.VECTOR_DB_BACKEND)
    await app.vectordb_client.connect()
    await app.vectordb_client.cache_connect()

    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG
    )

    app.cross_encoder = CrossEncoder(
        settings.RERANK_CROSS_ENCODER_NAME,
        model_kwargs={"dtype": dtype},
        trust_remote_code=True,
    )
    app.cross_encoder.to(device)
    
async def shutdown_span():
    await app.db_engine.dispose()
    await app.vectordb_client.disconnect()
    await app.vectordb_client.cache_disconnect()


app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base_router)
app.include_router(upload_router)
app.include_router(nlp_router)


# uvicorn main:app --reload --port 8001
