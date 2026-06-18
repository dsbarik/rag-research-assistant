import logging
import os
import shutil

from sqlmodel import Session, select

from src.config import settings
from src.config.settings import DOCUMENTS_DIR, LLM_MODEL_NAME
from src.conversation.manager import ConversationManager
from src.documents.schemas import Document
from src.documents.service import DocumentService
from src.infra.llm import OllamaLLM
from src.retrieval.service import RetrievalService

logger = logging.getLogger(__name__)


class OrchestratorService:
    def __init__(
        self,
        session: Session,
        doc_service: DocumentService,
        conv_manager: ConversationManager,
        retrieval_service: RetrievalService,
        llm: OllamaLLM
    ):
        self.session = session
        self.doc_service = doc_service
        self.conv_manager = conv_manager
        self.retrieval_service = retrieval_service
        self.llm = llm

    def ingest_file(self, file_stream, filename: str) -> dict:
        logger.info(f"Starting ingestion for file: {filename}")
        permanent_path = DOCUMENTS_DIR / filename

        try:
            with open(permanent_path, "xb") as buffer:
                shutil.copyfileobj(file_stream, buffer)
        except Exception as e:
            raise IOError(f"Failed to save file to storage: {str(e)}")

        try:
            doc = Document(
                path=str(permanent_path), doc_metadata={"filename": filename}
            )
            nodes = self.doc_service.ingest(doc)

            if not nodes:
                logger.warning(f"No chunks generated for file: {filename}")
                if permanent_path.exists():
                    os.remove(permanent_path)
                return {"status": "skipped", "document_id": 0, "chunks_processed": 0}

            self.retrieval_service.add_nodes(nodes)
            logger.info(
                f"Successfully ingested {len(nodes)} chunks for file: {filename}"
            )
            return {"status": "success", "document_id": doc.id, "chunks_processed": len(nodes)}

        except Exception as e:
            if permanent_path.exists():
                os.remove(permanent_path)
            raise e

    def chat(self, message: str, conversation_id: int | None) -> dict:
        """
        Flow: User Query -> Vector Search -> History Context -> LLM Answer -> Save to DB
        """
        logger.info(f"Processing chat message. Conversation ID: {conversation_id}")
        # 1. Manage Conversation
        if not conversation_id:
            conv = self.conv_manager.create_conversation(title=message[:30])
            conversation_id = conv.id

        if conversation_id is None:
            raise ValueError("Failed to create or retrieve conversation ID")

        # 2. Retrieve Context (delegated to RetrievalService)
        logger.info("Retrieving context via RetrievalService...")
        final_docs = self.retrieval_service.retrieve(message)

        context_str = "\n\n".join([doc["text"] for doc in final_docs])

        # 3. Get Conversation History
        history = self.conv_manager.get_context(conversation_id)

        # 4. Generate Answer (RAG)
        logger.info("Generating answer using LLM...")
        response_text = self.llm.generate_with_context(
            query=message,
            context=context_str,
            conversation_history=history,
            temperature=getattr(settings, "TEMPERATURE", 0.1),
        )

        # 5. Save Interaction
        self.conv_manager.add_message(conversation_id, "user", message)
        self.conv_manager.add_message(conversation_id, "assistant", response_text)

        return {
            "conversation_id": conversation_id,
            "response": response_text,
            "sources": [doc["metadata"] for doc in final_docs],
        }

    def list_documents(self):
        return self.doc_service.list_documents()

    def delete_document(self, doc_id: int):
        filename = self.doc_service.delete_document(doc_id)
        self.retrieval_service.delete_document(doc_id)
        return filename
