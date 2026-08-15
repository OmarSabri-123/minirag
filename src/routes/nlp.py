from tqdm.auto import tqdm
from logger import logger
from fastapi.responses import JSONResponse
from controllers import NLPController
from models import ResponseEnumeration
from models.ChunkModel import ChunkModel
from models.DomainModel import ProjectModel
from routes.schemes.nlp import PushRequest, SearchRequest
from fastapi import APIRouter, Request

nlp_router = APIRouter(
    prefix="/api/v1",
    tags=["multimodel-rag"]
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )
    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )
    project = await project_model.get_project_or_create_one(project_id=project_id)

    if not project:
        return JSONResponse(
            status_code=400,
            content={
                "signal": ResponseEnumeration.PROJECT_NOT_FOUND_ERROR.value
            }
        )

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        cross_encoder=request.app.cross_encoder,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

    collection_name = nlp_controller.create_collection_name(
        project_id=project.project_id)
    
    cache_name = nlp_controller.create_cache_name(
        project_id=project.project_id
    )

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset
    )
    
    _ = await request.app.vectordb_client.create_cache_collection(
        cache_name=cache_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset
    )

    total_chunks_count = await chunk_model.get_total_chunks_count(
        project_id=project.project_id
    )
    pbar = tqdm(total=total_chunks_count, desc="Vector Indexing", position=0)

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(project_id=project.project_id, page_no=page_no)
        if len(page_chunks):
            page_no += 1

        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        chunks_ids = [c.chunk_id for c in page_chunks]
        idx += len(page_chunks)

        is_inserted = await nlp_controller.index_into_vector_db(
            project=project,
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

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: int):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        cross_encoder=request.app.cross_encoder,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(
        project=project)

    return JSONResponse(
        status_code=200,
        content={
            "signal": ResponseEnumeration.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )


@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        cross_encoder=request.app.cross_encoder,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser
    )

    results = await nlp_controller.search_vector_db_collection(
        project=project,
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

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: int, search_request: SearchRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(project_id=project_id)

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        cross_encoder=request.app.cross_encoder,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser
    )

    query_vector = await nlp_controller.query_embeddings(
        text=search_request.text
    )

    # Retrieve answer from cache if exists
    cache_answer = await nlp_controller.retrieve_answer_from_cache(
        project=project,
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
        project=project,
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
        project=project,
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
