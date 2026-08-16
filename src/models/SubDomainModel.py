from .BaseDataModel import BaseDataModel
from .db_schemes import SubDomain
from sqlalchemy.future import select
from sqlalchemy import func

class SubDomainModel(BaseDataModel):

    def __init__(self, db_client):
        super().__init__(db_client)

    async def create_sub_domain(self, sub_domain: SubDomain):

        async with self.db_client() as session:
            async with session.begin():
                session.add(sub_domain)
            await session.commit()
            await session.refresh(sub_domain)
        return sub_domain

    async def get_sub_domain_by_name(self, domain_id: str, sub_domain_name: str):

        async with self.db_client() as session:
            async with session.begin():
                query = select(SubDomain).where(
                    SubDomain.domain_id == domain_id,
                    SubDomain.name == sub_domain_name
                )
                result = await session.execute(query)
                return result.scalar_one_or_none()


    async def get_sub_domain_or_create_one(self, domain_id: str, sub_domain_name: str,
                                           description: str=None):

        sub_domain = await self.get_sub_domain_by_name(
            domain_id=domain_id,
            sub_domain_name=sub_domain_name
        )
        if sub_domain is not None:
            return sub_domain

        sub_domain_rec = SubDomain(
            domain_id=domain_id,
            name=sub_domain_name,
            description=description
        )

        return await self.create_sub_domain(sub_domain=sub_domain_rec)

    async def get_all_sub_domains(self, domain_id: str, page: int=1, page_size: int=10):

        async with self.db_client() as session:
            async with session.begin():
                query = select(func.count(SubDomain.sub_domain_id)).where(
                    SubDomain.domain_id == domain_id
                )
                total_documents = await session.execute(query)
                total_documents = total_documents.scalar_one()

                total_pages = total_documents // page_size
                if total_documents % page_size > 0:
                    total_pages += 1

                query = select(SubDomain).where(
                    SubDomain.domain_id == domain_id
                ).offset((page - 1) * page_size).limit(page_size)
                result = await session.execute(query)
                sub_domains = result.scalars().all()

                return sub_domains, total_pages
