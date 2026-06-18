import datetime
import hashlib
import logging
import os
from pathlib import Path
from typing import List

from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import TextNode
from sqlmodel import Session, select

from src.config.settings import DOCUMENTS_DIR
from .loader import load_document
from .schemas import Document
from src.utils.exceptions import ServiceException

logger = logging.getLogger(__name__)


def compute_hash(path: str) -> str:
    """SHA-256 hash of a file for deduplication."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class DocumentService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def ingest(self, document: Document) -> List[TextNode]:
        """
        Ingest a document file into the system.
        """
        try:
            content_hash = compute_hash(document.path)

            existing = self.session.exec(
                select(Document).where(Document.hash == content_hash)
            ).first()

            if existing:
                logger.info(
                    f"Document '{document.path}' already ingested (hash match). Skipping."
                )
                return []

            # --- Load ---
            llama_docs = load_document(document.path)
            if not llama_docs:
                logger.warning(f"DoclingReader returned no content for '{document.path}'.")
                return []

            # --- Chunk with MarkdownNodeParser ---
            parser = MarkdownNodeParser()
            nodes: List[TextNode] = parser.get_nodes_from_documents(llama_docs)

            # --- Enrich metadata & save document record ---
            document.hash = content_hash
            if document.doc_metadata is None:
                document.doc_metadata = {}
            document.doc_metadata.update({
                "filename": Path(document.path).name,
                "page_count": llama_docs[0].metadata.get("page_count", 0),
                "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

            self.session.add(document)
            self.session.commit()
            self.session.refresh(document)

            # Attach document_id to every node so we can delete by document later
            for node in nodes:
                node.metadata["document_id"] = str(document.id)
                node.metadata.setdefault("filename", Path(document.path).name)

            logger.info(
                f"Ingested '{Path(document.path).name}': "
                f"{len(llama_docs)} pages → {len(nodes)} chunks."
            )
            return nodes

        except Exception as e:
            self.session.rollback()
            raise ServiceException(f"Failed to ingest document: {str(e)}") from e

    def list_documents(self):
        """Returns a list of all ingested documents."""
        statement = select(Document)
        docs = self.session.exec(statement).all()
        return [
            {
                "id": doc.id,
                "filename": doc.doc_metadata.get("filename", "Unknown"),
                "hash": doc.hash,
                "created_at": doc.doc_metadata.get("ingested_at"),
            }
            for doc in docs
        ]

    def delete_document(self, doc_id: int) -> str:
        """Deletes document from DB and physical storage."""
        doc = self.session.get(Document, doc_id)
        if not doc:
            raise ServiceException(f"Document with ID {doc_id} not found.")

        filename = doc.doc_metadata.get("filename")

        # Delete from DB
        self.session.delete(doc)
        self.session.commit()

        # Delete from file system
        if filename:
            file_path = DOCUMENTS_DIR / filename
            if file_path.exists():
                os.remove(file_path)

        return filename
