from .BaseDataModel import BaseDataModel
from .db_schemes import DataChunk, Document
from sqlalchemy.future import select
from sqlalchemy import func, delete
from typing import List

class ChunkModel(BaseDataModel):

    def __init__(self, db_client):
        super().__init__(db_client)

    async def create_chunk(self, chunk: DataChunk):
        async with self.db_client() as session: 
            async with session.begin():
                session.add(chunk)
            await session.commit()
            await session.refresh(chunk)
        return chunk
    
    async def get_chunk(self, chunk_id: str):
        async with self.db_client() as session: 
            async with session.begin():
                query = select(DataChunk).where(DataChunk.chunk_id == chunk_id)
                chunk = await session.execute(query) 
                chunk = chunk.scalar_one_or_none()
                return chunk
    
    async def insert_many_chunks(self, chunks: List[DataChunk], batch_size: int=100):
        async with self.db_client() as session: 
            async with session.begin():
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    session.add_all(batch)
            await session.commit()
        
        return len(chunks)
    
    
    async def delete_chunks_by_domain_id(self, domain_id: str):
        # chunks belong to documents, so the domain is reached through them
        async with self.db_client() as session:
            documents_query = select(Document.document_id).where(Document.domain_id == domain_id)
            query = delete(DataChunk).where(DataChunk.document_id.in_(documents_query))
            result = await session.execute(query)
            await session.commit()
        return result.rowcount

    async def delete_chunks_by_document_id(self, document_id: str):
        async with self.db_client() as session:
            query = delete(DataChunk).where(DataChunk.document_id == document_id)
            result = await session.execute(query)
            await session.commit()
        return result.rowcount

    async def get_domain_chunks(self, domain_id: str, page_no: int=1, page_size: int=50):
        async with self.db_client() as session:
            query = select(DataChunk).join(Document).where(
                Document.domain_id == domain_id
            ).order_by(
                DataChunk.document_id, DataChunk.chunk_index
            ).offset((page_no - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            records = result.scalars().all()
        return records

    async def get_total_chunks_count(self, domain_id: str):
        async with self.db_client() as session:
            query = select(func.count(DataChunk.chunk_id)).join(Document).where(
                Document.domain_id == domain_id
            )
            records_count = await session.execute(query)
            return records_count.scalar()

    async def get_filtered_chunks(self, domain_id: str, sub_domain_id: str = None,
                                  document_id: str = None, page_no: int = 1,
                                  page_size: int = 50):
        """Return one page of chunks limited to a domain, subdomain, or file."""
        conditions = [Document.domain_id == domain_id]
        if sub_domain_id is not None:
            conditions.append(Document.sub_domain_id == sub_domain_id)
        if document_id is not None:
            conditions.append(DataChunk.document_id == document_id)

        async with self.db_client() as session:
            query = select(DataChunk).join(Document).where(*conditions).order_by(
                DataChunk.document_id, DataChunk.chunk_index
            ).offset((page_no - 1) * page_size).limit(page_size)
            result = await session.execute(query)
            return result.scalars().all()

    async def get_filtered_chunks_count(self, domain_id: str, sub_domain_id: str = None,
                                        document_id: str = None):
        conditions = [Document.domain_id == domain_id]
        if sub_domain_id is not None:
            conditions.append(Document.sub_domain_id == sub_domain_id)
        if document_id is not None:
            conditions.append(DataChunk.document_id == document_id)

        async with self.db_client() as session:
            query = select(func.count(DataChunk.chunk_id)).join(Document).where(*conditions)
            result = await session.execute(query)
            return result.scalar()
    
