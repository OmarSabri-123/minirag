from pydantic import BaseModel
from typing import Optional

class PushRequest(BaseModel):
    do_reset: Optional[int] = 0
    domain_name: str
    sub_domain_name: str
    file_id: Optional[str] = None

class SearchRequest(BaseModel):
    text: str
    domain_name: str
    limit: Optional[int] = 5
