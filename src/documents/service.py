from typing import List
import hashlib
import datetime

from sqlmodel import Session, select

from .schemas import Document
from .loader import load_text
from .chunker import chunk_text


def compute_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class DocumentService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def ingest(self, document: Document) -> List[dict]:
        try:
            content_hash = compute_hash(document.path)

            existing = self.session.exec(
                select(Document).where(Document.hash == content_hash)
            ).first()

            if existing:
                return []

            text, derived_metadata = load_text(document.path)

            document.hash = content_hash

            if document.doc_metadata is None:
                document.doc_metadata = {}
            document.doc_metadata.update(derived_metadata)

            document.doc_metadata["ingested_at"] = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

            self.session.add(document)
            self.session.commit()
            self.session.refresh(document)

            chunks = chunk_text(text=text, document=document)
            
            return chunks

        except Exception as e:
            self.session.rollback()
            raise Exception(e)
