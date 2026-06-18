from typing import Any, Dict

from sqlalchemy import Column
from sqlalchemy.types import JSON
from sqlmodel import Field, SQLModel


class Document(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: int | None = Field(default=None, primary_key=True)

    path: str

    hash: str | None = Field(default=None, index=True)

    doc_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
