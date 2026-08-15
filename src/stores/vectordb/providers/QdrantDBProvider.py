from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMetricEnums, QdrantVectorType
from models.db_schemes import RetrievedDocument
from qdrant_client import AsyncQdrantClient, models
import uuid
from logger import logger
from typing import List

class QdrantDBProvider(VectorDBInterface):

    def __init__(self, db_url: str, api_key: str = None, distance_method: str = None,
                 cache_threshold=0.35, prefer_grpc: bool = False, timeout: int = 60):

        self.client = None
        self.cache_client = None
        self.db_url = db_url
        self.api_key = api_key
        self.prefer_grpc = prefer_grpc
        self.timeout = timeout
        self.distance_method = None
        self.modifier = models.Modifier.IDF
        self.cache_threshold = cache_threshold

        if distance_method == DistanceMetricEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMetricEnums.EUCLIDEAN.value:
            self.distance_method = models.Distance.EUCLID
        elif distance_method == DistanceMetricEnums.DOT_PRODUCT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logger

    def _new_client(self) -> AsyncQdrantClient:
        return AsyncQdrantClient(
            url=self.db_url,
            api_key=self.api_key,
            prefer_grpc=self.prefer_grpc,
            timeout=self.timeout
        )

    async def cache_connect(self):
        self.cache_client = self._new_client()
        self.logger.info(f"Connected Qdrant cache client to {self.db_url}")

    async def cache_disconnect(self):
        if self.cache_client is not None:
            await self.cache_client.close()
        self.cache_client = None

    async def is_cache_collection_exists(self, cache_name: str) -> bool:
        return await self.cache_client.collection_exists(collection_name=cache_name)

    async def delete_cache_collection(self, cache_name: str) -> bool:
        if await self.is_cache_collection_exists(cache_name):
            await self.cache_client.delete_collection(collection_name=cache_name)
            return True
        return False

    async def create_cache_collection(self, cache_name: str, embedding_size: int, do_reset: bool = False):

        if do_reset:
            _ = await self.delete_cache_collection(cache_name)

        if not await self.is_cache_collection_exists(cache_name):
            self.logger.info(
                f"Creating new Qdrant cache collection: {cache_name}")

            await self.cache_client.create_collection(
                collection_name=cache_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=models.Distance.EUCLID
                )
            )

    async def search_cache(self, cache_name: str, vector: list):

        if not await self.is_cache_collection_exists(cache_name):
            return []

        search_result = await self.cache_client.query_points(
            collection_name=cache_name,
            query=vector,
            limit=1
        )
        return search_result.points

    async def add_to_cache(self, cache_name: str, vector: list, response_text: str):

        point_id = str(uuid.uuid4())
        point = models.PointStruct(
            id=point_id,
            vector=vector,
            payload={
                "response_text": response_text
            }
        )
        try:
            await self.cache_client.upsert(
                collection_name=cache_name,
                points=[point]
            )
        except Exception as e:
            logger.error(f"Error while inserting cache: {e}")
            return False

        return True

    async def connect(self):
        self.client = self._new_client()
        self.logger.info(f"Connected Qdrant client to {self.db_url}")

    async def disconnect(self):
        if self.client is not None:
            await self.client.close()
        self.client = None

    async def is_collection_exists(self, collection_name: str) -> bool:
        return await self.client.collection_exists(collection_name=collection_name)

    async def list_all_collections(self):
        return await self.client.get_collections()

    async def get_collection_info(self, collection_name: str) -> dict:
        return await self.client.get_collection(collection_name=collection_name)

    async def delete_collection(self, collection_name: str) -> bool:
        if await self.is_collection_exists(collection_name):
            await self.client.delete_collection(collection_name=collection_name)
            return True
        return False

    async def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):

        if do_reset:
            _ = await self.delete_collection(collection_name)

        if not await self.is_collection_exists(collection_name):
            self.logger.info(
                f"Creating new Qdrant collection: {collection_name}")

            await self.client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    QdrantVectorType.DENSE.value: models.VectorParams(
                        size=embedding_size,
                        distance=self.distance_method
                    ),
                },
                sparse_vectors_config={
                    QdrantVectorType.SPARSE.value: models.SparseVectorParams(
                        modifier=self.modifier
                    )
                }
            )

            return True

        return False

    async def insert_one(self, collection_name: str, text: str, vector: list, metadata: dict = None, id: str = None):

        if not await self.is_collection_exists(collection_name):
            logger.error(f"Collection {collection_name} does not exist.")
            return False

        point = models.PointStruct(
            id=id,
            vector=vector,
            payload={
                "text": text,
                "metadata": metadata
            }
        )

        await self.client.upsert(
            collection_name=collection_name,
            points=[point]
        )

        return True

    async def insert_many(self, collection_name: str, texts: list, vectors: list, metadatas: list = None, ids: list = None, batch_size: int = 50):

        if not await self.is_collection_exists(collection_name):
            logger.error(f"Collection {collection_name} does not exist.")
            return False

        for start_idx in range(0, len(texts), batch_size):

            batch_text = texts[start_idx: start_idx + batch_size]
            batch_vectors = vectors[start_idx: start_idx + batch_size]
            batch_metadatas = metadatas[start_idx: start_idx + batch_size]
            batch_ids = ids[start_idx: start_idx + batch_size]

            batch_points = [
                models.PointStruct(
                    id=batch_ids[x],
                    vector={
                        QdrantVectorType.DENSE.value: batch_vectors[x],
                        QdrantVectorType.SPARSE.value: models.Document(
                            text=batch_text[x],
                            model="Qdrant/bm25",
                        ),
                    },
                    payload={
                        "text": batch_text[x],
                        "metadata": batch_metadatas[x]
                    }
                )
                for x in range(len(batch_text))
            ]

            try:
                await self.client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )

            except Exception as e:
                logger.error(f"Error while inserting batch: {e}")
                return False

        return True

    async def search_by_vector(self, collection_name: str, text: str, query_vector: List, limit: int):

        results = await self.client.query_points(
            collection_name=collection_name,
            prefetch=[
                models.Prefetch(
                    query=query_vector,
                    using=QdrantVectorType.DENSE.value,
                    limit=limit,
                ),
                models.Prefetch(
                    query=models.Document(
                        text=text,
                        model="Qdrant/bm25",
                    ),
                    using=QdrantVectorType.SPARSE.value,
                    limit=limit,
                )
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.DBSF
            ),
            limit=limit
        )

        results = [
            RetrievedDocument(**{
                "score": res.score,
                "text": res.payload["text"]
            })
            for res in results.points
        ]

        return results
