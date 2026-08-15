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
    
    async def get_all_domain_projects(self, asset_project_id: str, asset_type: str):
        async with self.db_client() as session:
            stmt = select(Document).where(
                Document.asset_project_id == asset_project_id,
                Document.asset_type == asset_type
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
        return records
    
    async def get_document_record(self, asset_project_id: str, asset_name: str):
        async with self.db_client() as session: 
            async with session.begin():
                query = select(Document).where(
                    Document.asset_project_id == asset_project_id,
                    Document.asset_name == asset_name
                    )
                result = await session.execute(query) 
                result = result.scalar_one_or_none()
                return result
    
