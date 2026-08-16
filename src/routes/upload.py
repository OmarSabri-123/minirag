import os
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from logger import setup_logger
from models import ResponseEnumeration
from models.ChunkModel import ChunkModel
from models.DomainModel import DomainModel
from models.DocumentModel import DocumentModel
from models.SubDomainModel import SubDomainModel
from models.db_schemes import Document, DataChunk
from controllers import UploadController, ProcessController, NLPController
from routes import ProcessRequest, UploadRequest

upload_router = APIRouter(
    prefix="/api/v1",
    tags=["Data-Ingestion"],
)

logger = setup_logger(name="uvicorn")

@upload_router.post("/upload")
async def upload_file(request: Request, upload: UploadRequest = Depends(UploadRequest.as_upload),
                      app_config: Settings = Depends(get_settings)):

    domain_model = await DomainModel.create_instance(db_client=request.app.db_client)
    sub_domain_model = await SubDomainModel.create_instance(db_client=request.app.db_client)
    

    domain = await domain_model.get_domain_or_create_one(domain_name=upload.domain_name)
    sub_domain = await sub_domain_model.get_sub_domain_or_create_one(
        domain_id=domain.domain_id,
        sub_domain_name=upload.sub_domain_name
    )

    logger.info(f"Received file upload request for domain_name: {upload.domain_name}, filename: {upload.file.filename}")

     # Validate file

    upload_object = UploadController()

    is_valid, message = upload_object.validate_file(file=upload.file)

    if not is_valid:
        logger.error(f"File validation failed: {message}")
        return JSONResponse(status_code=400, content={"signal": message})

    source_type, message = upload_object.get_file_type(original_filename=upload.file.filename)

    if source_type is None:
        logger.error(f"File type detection failed: {message}")
        return JSONResponse(status_code=400, content={"signal": message})

    # Generate unique filename and save file

    file_location, file_id=upload_object.generate_unique_filename(
        original_filename=upload.file.filename,
        project_id=sub_domain.sub_domain_id,
        )

    try:
        content_hash = await upload_object.write_file_and_get_hash(
            file=upload.file,
            file_location=file_location
        )
        logger.info(f"File saved successfully at {file_location}")

    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return JSONResponse(status_code=400, content={"signal": ResponseEnumeration.FILE_UPLOAD_FAILED.value})

    language = await upload_object.detect_file_language(
        file_location=file_location,
        source_type=source_type
    )

    document_model=await DocumentModel.create_instance(
        db_client=request.app.db_client
    )

    document_resource = Document(
        domain_id=domain.domain_id,
        sub_domain_id=sub_domain.sub_domain_id,
        title=upload.file.filename,
        source_name=file_id,
        source_type=source_type,
        language=language,
        content_hash=content_hash
    )

    document_record = await document_model.create_document(document=document_resource)

    return JSONResponse(
            status_code = 200,
            content={
                "signal": ResponseEnumeration.FILE_UPLOADED_SUCCESS.value,
                "file_id": str(document_record.document_id),
            }
        )

@upload_router.post("/process")
async def process_endpoint(request: Request, process_request: ProcessRequest):

    do_reset = process_request.do_reset

    domain_model = await DomainModel.create_instance(db_client=request.app.db_client)

    domain = await domain_model.get_domain_by_name(domain_name=process_request.domain_name) 

    if not domain:
        return JSONResponse(
            status_code = 400,
            content = {
                "signal": ResponseEnumeration.PROJECT_NOT_FOUND_ERROR.value
            }
        )

    sub_domain_model = await SubDomainModel.create_instance(
        db_client=request.app.db_client
    )
    
    document_model = await DocumentModel.create_instance(
        db_client=request.app.db_client
    )
    
    sub_domain = await sub_domain_model.get_sub_domain_by_name(
        domain_id=domain.domain_id,
        sub_domain_name=process_request.sub_domain_name
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser
    )

    if process_request.file_id:
        document_record = await document_model.get_document_by_id(
            document_id=process_request.file_id,
            domain_id = domain.domain_id,
            sub_domain_id = sub_domain.sub_domain_id,
        )

        if not document_record:
            return JSONResponse(
                status_code = 400,
                content = {
                    "signal":ResponseEnumeration.FILE_ID_ERROR.value
                }
            )

        domain_documents = [document_record]

    else:
        domain_documents = await document_model.get_all_domain_documents(
            domain_id = domain.domain_id,
            sub_domain_id = sub_domain.sub_domain_id
        )

    if len(domain_documents) == 0:
        return JSONResponse(
            status_code = 400,
            content={
                "signal": ResponseEnumeration.NO_FILES_ERROR.value,
            }
        )

    no_records = 0
    no_files = 0

    if do_reset:
        _ = await nlp_controller.reset_vector_db_collection(
            domain=domain
        )
        _ = await chunk_model.delete_chunks_by_domain_id(
            domain_id=domain.domain_id
        )

    for document_record in domain_documents:

        # files are stored under their sub domain, so the process path has to
        # match the one used at upload time
        process_controller = ProcessController(project_id=document_record.sub_domain_id)

        chunks = process_controller.load_and_export(
            file_name=document_record.source_name
        )

        if chunks is None or len(chunks) == 0:
            return JSONResponse(
                status_code = 400,
                content={
                    "signal": ResponseEnumeration.PROCESSING_FAILED.value
                }
            )

        file_chunks_records = [
            DataChunk(
                document_id=document_record.document_id,
                chunk_index=i+1,
                content=chunk.page_content,
                chunk_metadata={
                    **chunk.metadata,
                    "domain_name": domain.name,
                    "sub_domain_name": sub_domain.name,
                    "document_id": str(document_record.document_id),
                },
            )
            for i, chunk in enumerate(chunks)
        ]

        no_records += await chunk_model.insert_many_chunks(chunks=file_chunks_records)
        no_files += 1
    return JSONResponse(
        content={
            "signal": ResponseEnumeration.PROCESSING_SUCCESS.value,
            "inserted_chunks": no_records,
            "processed_files": no_files
        }
    )
