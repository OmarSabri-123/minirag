from .BaseDataModel import BaseDataModel
from .db_schemes import Document
from sqlalchemy.future import select

class DocumentModel(BaseDataModel):
    
    def __init__(self, db_client):
        super().__init__(db_client)

    async def create_document(self, document: Document):
        async with self.db_client() as session: 
            async with session.begin():
                session.add(document)
            await session.commit()
            await session.refresh(document)
        return document     
    
    async def update_document_language(self, document_id: str, language: str):
        async with self.db_client() as session:
            async with session.begin():
                query = select(Document).where(Document.document_id == document_id)
                result = await session.execute(query)
                document = result.scalar_one_or_none()

                if document is None:
                    return None

                document.language = language

            await session.commit()
            await session.refresh(document)
        return document

    async def get_all_domain_documents(self, domain_id: str, sub_domain_id: str=None):
        async with self.db_client() as session:
            stmt = select(Document).where(Document.domain_id == domain_id)

            if sub_domain_id is not None:
                stmt = stmt.where(Document.sub_domain_id == sub_domain_id)

            result = await session.execute(stmt)
            records = result.scalars().all()
        return records

    async def get_document_record(self, domain_id: str, source_name: str,
                                  sub_domain_id: str = None):
        async with self.db_client() as session:
            async with session.begin():
                conditions = [
                    Document.domain_id == domain_id,
                    Document.source_name == source_name,
                ]
                if sub_domain_id is not None:
                    conditions.append(Document.sub_domain_id == sub_domain_id)

                query = select(Document).where(*conditions)
                result = await session.execute(query)
                result = result.scalar_one_or_none()
                return result

    async def get_document_by_id(self, document_id: str, domain_id: str = None,
                                 sub_domain_id: str = None):
        async with self.db_client() as session:
            conditions = [Document.document_id == document_id]
            if domain_id is not None:
                conditions.append(Document.domain_id == domain_id)
            if sub_domain_id is not None:
                conditions.append(Document.sub_domain_id == sub_domain_id)

            result = await session.execute(select(Document).where(*conditions))
            return result.scalar_one_or_none()

    async def get_document_by_hash(self, content_hash: str):
        async with self.db_client() as session:
            async with session.begin():
                query = select(Document).where(Document.content_hash == content_hash)
                result = await session.execute(query)
                return result.scalar_one_or_none()
    
