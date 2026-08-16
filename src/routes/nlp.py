from tqdm.auto import tqdm
from logger import logger
from fastapi.responses import JSONResponse
from controllers import NLPController
from models import ResponseEnumeration
from models.ChunkModel import ChunkModel
from models.DomainModel import DomainModel
from models.DocumentModel import DocumentModel
from models.SubDomainModel import SubDomainModel
from routes.schemes.nlp import PushRequest, SearchRequest
from fastapi import APIRouter, Request

nlp_router = APIRouter(
    prefix="/api/v1",
    tags=["Search-RAG"],
)

@nlp_router.post("/index/push")
async def index_project(request: Request, push_request: PushRequest):

    domain_model = await DomainModel.create_instance(
        db_client=request.app.db_client
    )
    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )
    document_model = await DocumentModel.create_instance(
        db_client=request.app.db_client
    )
    domain = await domain_model.get_domain_by_name(
        domain_name=push_request.domain_name)

    if not domain:
        return JSONResponse(
            status_code=400,
            content={
                "signal": ResponseEnumeration.PROJECT_NOT_FOUND_ERROR.value
            }
        )

    sub_domain_model = await SubDomainModel.create_instance(
        db_client=request.app.db_client
    )
    sub_domain = await sub_domain_model.get_sub_domain_by_name(
        domain_id=domain.domain_id,
        sub_domain_name=push_request.sub_domain_name,
    )
    if not sub_domain:
        return JSONResponse(
            status_code=400,
            content={"signal": ResponseEnumeration.PROJECT_NOT_FOUND_ERROR.value},
        )

    document = None
    if push_request.file_id is not None:
        document = await document_model.get_document_by_id(
            document_id=push_request.file_id,
            domain_id=domain.domain_id,
            sub_domain_id=sub_domain.sub_domain_id,
        )
        if not document:
            return JSONResponse(
                status_code=400,
                content={"signal": ResponseEnumeration.FILE_ID_ERROR.value},
            )

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser,
        cross_encoder=getattr(request.app, "cross_encoder", None)
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    collection_name = nlp_controller.create_collection_name(
        domain_id=domain.domain_id)

    # cache_name = nlp_controller.create_cache_name(
    #     domain_id=domain.domain_id
    # )

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset
    )
    
    # _ = await request.app.vectordb_client.create_cache_collection(
    #     cache_name=cache_name,
    #     embedding_size=request.app.embedding_client.embedding_size,
    #     do_reset=push_request.do_reset
    # )

    total_chunks_count = await chunk_model.get_filtered_chunks_count(
        domain_id=domain.domain_id,
        sub_domain_id=sub_domain.sub_domain_id,
        document_id=document.document_id if document else None,
    )
    pbar = tqdm(total=total_chunks_count, desc="Vector Indexing", position=0)

    while has_records:
        page_chunks = await chunk_model.get_filtered_chunks(
            domain_id=domain.domain_id,
            sub_domain_id=sub_domain.sub_domain_id,
            document_id=document.document_id if document else None,
            page_no=page_no,
        )

        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids = [c.chunk_id for c in page_chunks]
        page_no += 1
        idx += len(page_chunks)

        is_inserted = await nlp_controller.index_into_vector_db(
            domain=domain,
            chunks=page_chunks,
            chunks_ids=chunks_ids
        )

        if not is_inserted:
            return JSONResponse(
                status_code=400,
                content={
                    "signal": ResponseEnumeration.INSERT_INTO_VECTORDB_ERROR.value
                }
            )
        pbar.update(len(page_chunks))
        inserted_items_count += len(page_chunks)

    return JSONResponse(
        status_code=200,
        content={
            "signal": ResponseEnumeration.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{domain_name}")
async def get_project_index_info(request: Request, domain_name: str):

    domain_model = await DomainModel.create_instance(
        db_client=request.app.db_client
    )

    domain = await domain_model.get_domain_by_name(domain_name=domain_name)

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser,
        cross_encoder=getattr(request.app, "cross_encoder", None)
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(
        domain=domain)

    return JSONResponse(
        status_code=200,
        content={
            "signal": ResponseEnumeration.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )


@nlp_router.post("/index/search/")
async def search_index(request: Request, search_request: SearchRequest):

    domain_model = await DomainModel.create_instance(
        db_client=request.app.db_client
    )

    domain = await domain_model.get_domain_by_name(domain_name=search_request.domain_name)

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser,
        cross_encoder=getattr(request.app, "cross_encoder", None)
    )

    results = await nlp_controller.search_vector_db_collection(
        domain=domain,
        query=search_request.text,
        limit=search_request.limit
    )

    if not results:
        return JSONResponse(
            status_code=400,
            content={
                "signal": ResponseEnumeration.VECTORDB_SEARCH_ERROR.value
            }
        )

    return JSONResponse(
        status_code=200,
        content={
            "signal": ResponseEnumeration.VECTORDB_SEARCH_SUCCESS.value,
            "results": results
        }
    )

@nlp_router.post("/index/answer/{domain_name}")
async def answer_rag(request: Request, domain_name: str, search_request: SearchRequest):

    domain_model = await DomainModel.create_instance(
        db_client=request.app.db_client
    )

    domain = await domain_model.get_domain_by_name(domain_name=domain_name)

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser,
        cross_encoder=getattr(request.app, "cross_encoder", None)
    )

    query_vector = await nlp_controller.query_embeddings(
        text=search_request.text
    )

    # Retrieve answer from cache if exists
    cache_answer = await nlp_controller.retrieve_answer_from_cache(
        domain=domain,
        query_vector=query_vector
    )

    if cache_answer:
        return JSONResponse(
            status_code=200,
            content={
                "signal": ResponseEnumeration.CACHE_ANSWER_SUCCESS.value,
                "answer_from_cache": cache_answer
            }
        )

    answer, full_prompt, chat_history = await nlp_controller.rag_answer_question(
        domain=domain,
        query=search_request.text,
        limit=search_request.limit
    )

    if not answer:
        return JSONResponse(
            status_code=400,
            content={
                "signal": ResponseEnumeration.RAG_ANSWER_ERROR.value
            }
        )
    
    _ = await nlp_controller.add_answer_into_cache(
        domain=domain,
        query_vector=query_vector,
        answer=answer
    )

    return JSONResponse(
        status_code=200,
        content={
            "signal": ResponseEnumeration.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history

        }
    )
