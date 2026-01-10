from typing import Dict, Any
from sqlmodel import SQLModel, Field
from sqlalchemy.types import JSON
from sqlalchemy import Column


class Document(SQLModel, table=True):
    
    id: int | None = Field(
        default=None,
        primary_key=True)
    
    path: str
    
    hash: str | None = Field(
        default=None,
        index=True
    )
    
    doc_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


