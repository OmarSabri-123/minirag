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
    
    async def get_domain_or_create_one(self, domain_id: str):

        async with self.db_client() as session:
            async with session.begin():
                query = select(Domain).where(Domain.domain_id == domain_id)
                result = await session.execute(query)
                domain = result.scalar_one_or_none()
                if domain is None:
                    domain_rec = Domain(
                        domain_id = domain_id
                    )
                    domain = await self.create_domain(domain=domain_rec)
                    return domain
                else:
                    return domain
    
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
            



