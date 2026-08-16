from .BaseDataModel import BaseDataModel
from .db_schemes import Domain
from sqlalchemy.future import select
from sqlalchemy import func

class DomainModel(BaseDataModel):
    
    def __init__(self, db_client):
        super().__init__(db_client)

    async def create_domain(self, domain: Domain):

        async with self.db_client() as session: 
            async with session.begin():
                session.add(domain)
            await session.commit()
            await session.refresh(domain)
        return domain
    
    async def get_domain_by_name(self, domain_name: str):

        async with self.db_client() as session:
            async with session.begin():
                query = select(Domain).where(Domain.name == domain_name)
                result = await session.execute(query)
                return result.scalar_one_or_none()
            
    
    async def get_domain_or_create_one(self, domain_name: str, description: str=None):

        domain = await self.get_domain_by_name(domain_name=domain_name)
        if domain is not None:
            return domain

        domain_rec = Domain(
            name=domain_name,
            description=description
        )

        return await self.create_domain(domain=domain_rec)
    
    async def get_all_domains(self, page: int=1, page_size: int=10):

        async with self.db_client() as session:
            async with session.begin():
                query = select(func.count(Domain.domain_id))
                total_documents = await session.execute(query)
                total_documents = total_documents.scalar_one()

                total_pages = total_documents // page_size
                if total_documents % page_size > 0:
                    total_pages += 1
                
                query = select(Domain).offset((page - 1) * page_size).limit(page_size)
                domains = await session.execute(query).scalars().all()

                return domains, total_pages
            



