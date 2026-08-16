from pydantic import BaseModel
from typing import Optional
from fastapi import File, Form, UploadFile

class UploadRequest(BaseModel):
    domain_name: str
    sub_domain_name: Optional[str] = "default"
    file: UploadFile

    @classmethod
    def as_upload(
        cls,
        domain_name: str = Form(...),
        sub_domain_name: Optional[str] = Form("default"),
        file: UploadFile = File(...)
    ):
        return cls(
            domain_name=domain_name,
            sub_domain_name=sub_domain_name,
            file=file
        )
    
    

