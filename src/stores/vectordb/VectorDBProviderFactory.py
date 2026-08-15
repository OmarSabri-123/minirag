from .providers import QdrantDBProvider
from .providers import PGVectorProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController
from sqlalchemy.orm import sessionmaker

class VectorDBProviderFactory:

    def __init__(self, config: dict, db_client: sessionmaker = None):
        self.config = config
        self.base_controller = BaseController()
        self.db_client = db_client
    
    def create(self, provider: str):

        if provider == VectorDBEnums.QDRANT.value:

            return QdrantDBProvider(
                db_url = self.config.QDRANT_URL,
                api_key = self.config.QDRANT_API_KEY,
                distance_method = self.config.VECTOR_DB_DISTANCE_METHOD,
                prefer_grpc = self.config.QDRANT_PREFER_GRPC
            )
        
        if provider == VectorDBEnums.PGVECTOR.value:

            return PGVectorProvider(
                db_client = self.db_client,
                default_vector_size = self.config.EMBEDDING_MODEL_DIMENSION,
                distance_method = self.config.VECTOR_DB_DISTANCE_METHOD,
                index_threshold = self.config.INDEX_THRESHOLD
            )
        
        return None

