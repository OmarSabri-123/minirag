from .BaseController import BaseController
from stores.llm.LLMEnums import DocumentTypeEnum
from models.db_schemes import Domain, DataChunk
from routes.schemes.QueryExpand import SemanticExpansion
from typing import List
import json

class NLPController(BaseController):

    def __init__(self, vector_db_client, embedding_client, generation_client, template_parser,
                 cross_encoder=None):
        super().__init__()
        self.vector_db_client = vector_db_client
        self.cross_encoder = cross_encoder
        self.embedding_client = embedding_client
        self.generation_client = generation_client
        self.template_parser = template_parser

    def create_collection_name(self, domain_id: str):
        return f"collection_{domain_id}".strip()
    
    def create_cache_name(self, domain_id: str):
        return f"cache_{domain_id}".strip()

    async def reset_vector_db_collection(self, domain: Domain):
        collection_name = self.create_collection_name(
            domain_id=domain.domain_id)
        return await self.vector_db_client.delete_collection(collection_name=collection_name)

    async def get_vector_db_collection_info(self, domain: Domain):
        collection_name = self.create_collection_name(
            domain_id=domain.domain_id)
        collection_info = await self.vector_db_client.get_collection_info(
            collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )

    async def index_into_vector_db(self, domain: Domain, chunks: List[DataChunk],
                             chunks_ids: List[int],
                             do_reset: bool = False):

        collection_name = self.create_collection_name(
            domain_id=domain.domain_id)

        texts = [c.content for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]
        vectors = await self.embedding_client.embed_text(
            texts, DocumentTypeEnum.DOCUMENT.value)

        _ = await self.vector_db_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset)

        _ = await self.vector_db_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadatas=metadata,
            ids=chunks_ids
        )

        return True
    
    async def query_expansion(self, query:str):

        system_prompt = self.template_parser.get("rag", "query_expand_system_prompt")
        user_prompt = self.template_parser.get("rag", "query_expand_user_prompt", {
            "query": query
        })

        chat_history = [
            self.generation_client.construt_prompt(
                prompt = system_prompt,
                role = self.generation_client.enums.SYSTEM.value
            )
        ]

        answer = await self.generation_client.generate_text(
            prompt = user_prompt,
            chat_history = chat_history
        )

        if not answer:
            return False

        return SemanticExpansion(
            original_query = query,
            expanded_query = answer
        )
    
    async def query_embeddings(self, text: str):
        
        vectors = await self.embedding_client.embed_text(
            text, DocumentTypeEnum.QUERY.value)

        if not vectors or len(vectors) == 0:
            return False

        if isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]

        if not query_vector:
            return False
        
        return query_vector
    
    async def retrieve_answer_from_cache(self, domain: Domain, query_vector: list, cache_threshold=0.7):

        cache_name = self.create_cache_name(
            domain_id=domain.domain_id
        )

        cache_result = await self.vector_db_client.search_cache(
            cache_name=cache_name,
            vector=query_vector
        )
        if cache_result:
            for s in cache_result:
                if s.score <= cache_threshold:
                    return s.payload["response_text"]
    
    async def add_answer_into_cache(self, domain: Domain, query_vector: list, answer: str):

        cache_name = self.create_cache_name(
            domain_id=domain.domain_id
        )

        _ = await self.vector_db_client.add_to_cache(
            cache_name=cache_name,
            vector=query_vector,
            response_text=answer
        )
        return True
    
    async def rerank_documents(self, expanded_query: str, documents: list):

        # the cross encoder is optional, without it the vector search order is kept
        if self.cross_encoder is None:
            return [{"text": document, "score": None} for document in documents[:3]]

        rankings = self.cross_encoder.rank(
            expanded_query,
            documents,
            return_documents=True,
            convert_to_tensor=True
        )
        result = [
            {
                "text": ranking['text'],
                "score": f"{ranking['score']:.4f}"
            }
            for ranking in rankings[:3]
        ]
        return result

    async def search_vector_db_collection(self, domain: Domain, query: str, limit: int = 5):

        query_optimization = await self.query_expansion(
            query=query
        )

        if not query_optimization or not query_optimization.expanded_query:
           return False
        
        expanded_query_vector = await self.query_embeddings(
            text=query_optimization.expanded_query
        )
        
        collection_name = self.create_collection_name(
            domain_id=domain.domain_id)

        result = await self.vector_db_client.search_by_vector(
            collection_name=collection_name,
            text=query_optimization.expanded_query,
            query_vector=expanded_query_vector,
            limit=limit
        )

        if not result:
            return False
        
        documents = [res.text for res in result]

        result_rerank = await self.rerank_documents(
        expanded_query=query_optimization.expanded_query,
        documents=documents
        )

        return result_rerank
    
    async def rag_answer_question(self, domain: Domain, query:str, limit: int = 5):

        answer, full_prompt, chat_history = None, None, None

        retrieved_documents = await self.search_vector_db_collection(
            domain=domain,
            query=query,
            limit=limit
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history

        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompt = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                "doc_num": idx+1,
                "chunk_text": doc['text']
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        full_prompt = "\n\n".join([documents_prompt, footer_prompt])

        chat_history = [
            self.generation_client.construt_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value
            )
        ]

        answer = await self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history
