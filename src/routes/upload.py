import os
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse
from helpers.config import get_settings, Settings
from logger import setup_logger
from models import ResponseEnumeration, AssetTypeEnum
from models.ChunkModel import ChunkModel
from models.DomainModel import ProjectModel
from models.AssetModel import AssetModel
from models.db_schemes import Asset, DataChunk
from controllers import UploadController, ProcessController, NLPController
from routes import ProcessRequest, UploadRequest
import aiofiles

upload_router = APIRouter(
    prefix="/api/v1",
    tags=["multimodel-rag"]
)

logger = setup_logger(name="uvicorn")

@upload_router.post("/upload/{project_id}")
async def upload_file(request: Request, upload: UploadRequest = Depends(UploadRequest.as_upload), 
                      app_config: Settings = Depends(get_settings)):

    project_model = await ProjectModel.create_instance(db_client = request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id = upload.project_id)

    logger.info(f"Received file upload request for project_id: {upload.project_id}, filename: {upload.file.filename}")

     # Validate file

    upload_object = UploadController()

    is_valid, message = upload_object.validate_file(file = upload.file)

    if not is_valid:
        logger.error(f"File validation failed: {message}")
        return JSONResponse(status_code=400, content={"signal": message})
    
    # Generate unique filename and save file

    file_location, file_id = upload_object.generate_unique_filename(
        original_filename = upload.file.filename, 
        project_id = upload.project_id
        )

    try:
        async with aiofiles.open(file_location, 'wb') as f:
            while content := await upload.file.read(app_config.FILE_DEFAULT_CHUNK_SIZE):  # Read file in chunks
                await f.write(content)
        logger.info(f"File saved successfully at {file_location}")
        
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        return JSONResponse(status_code=400, content={"signal": ResponseEnumeration.FILE_UPLOAD_FAILED.value})
    
    asset_model = await AssetModel.create_instance(
        db_client = request.app.db_client
        )

    asset_resource = Asset(
        asset_project_id = project.project_id,
        asset_type = AssetTypeEnum.FILE.value,
        asset_name = file_id,
        asset_size = os.path.getsize(file_location)
    )

    asset_record = await asset_model.create_asset(asset = asset_resource)

    return JSONResponse(
            status_code = 200,
            content={
                "signal": ResponseEnumeration.FILE_UPLOADED_SUCCESS.value,
                "file_id": str(asset_record.asset_id),
            }
        )

@upload_router.post("/process/{project_id}")
async def process_endpoint(request: Request, project_id: int, process_request: ProcessRequest):

    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap
    do_reset = process_request.do_reset

    project_model = await ProjectModel.create_instance(db_client = request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id = project_id)

    asset_model = await AssetModel.create_instance(
        db_client = request.app.db_client
        )
    
    chunk_model = await ChunkModel.create_instance(
        db_client = request.app.db_client
    )
    
    nlp_controller = NLPController(
        vector_db_client=request.app.vectordb_client,
        embedding_client=request.app.embedding_client,
        generation_client=request.app.generation_client,
        template_parser=request.app.template_parser
    )

    project_files_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id = project.project_id,
            asset_name = process_request.file_id
        )

        if not asset_record:
            return JSONResponse(
                status_code = 400,
                content = {
                    "signal":ResponseEnumeration.FILE_ID_ERROR.value
                }
            )
        
        project_files_ids = {
            asset_record.asset_id: asset_record.asset_name
        }
    
    else:
        asset_files = await asset_model.get_all_project_assets(
            asset_project_id = project.project_id,
            asset_type = AssetTypeEnum.FILE.value
        )

        project_files_ids = {
            asset_record.asset_id: asset_record.asset_name
            for asset_record in asset_files
        }

    
    if len(project_files_ids) == 0:
        return JSONResponse(
            status_code = 400,
            content={
                "signal": ResponseEnumeration.NO_FILES_ERROR.value,
            }
        )
    
    process_controller = ProcessController(project_id=project_id)
    
    no_records = 0
    no_files = 0

    if do_reset:
        nlp_controller = nlp_controller.reset_vector_db_collection(
            project = project
        )
        _ = await chunk_model.delete_chunks_by_project_id(
            project_id = project.project_id
        )
    
    for asset_id, asset_name in project_files_ids.items():

        chunks = process_controller.load_and_export(
            file_name = asset_name
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
                chunk_text=chunk.page_content,
                chunk_metadata=chunk.metadata,
                chunk_order=i+1,
                chunk_project_id=project.project_id,
                chunk_asset_id=asset_id
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
