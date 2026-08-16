from pydantic import BaseModel, Field
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: Optional[str] = None
    chunk_size: Optional[int] = 100
    overlap: Optional[int] = 20
    do_reset: Optional[int] = 0
    domain_name: str
    sub_domain_name: Optional[str] = None
