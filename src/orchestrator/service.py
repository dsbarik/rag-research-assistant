import os
from sqlmodel import Session, select
import shutil

from src.conversation.manager import ConversationManager
from src.documents.schemas import Document
from src.documents.service import DocumentService
from src.infra.llm import OllamaLLM
from src.infra.vectorstore.chroma import ChromaVectorStore
from src.config import DOCUMENTS_DIR, LLM_MODEL_NAME


class OrchestratorService:
    def __init__(self, session: Session):
        self.session = session
        self.doc_service = DocumentService(session)
        self.conv_manager = ConversationManager(session)

        self.vector_store = ChromaVectorStore()
        self.llm = OllamaLLM(model=LLM_MODEL_NAME)

    def ingest_file(self, file_stream, filename: str):
        permanent_path = DOCUMENTS_DIR / filename
        
        try:
            # 1. Save the stream to our permanent storage
            with open(permanent_path, "wb") as buffer:
                shutil.copyfileobj(file_stream, buffer)
        except Exception as e:
            raise IOError(f"Failed to save file to storage: {str(e)}")
        
        # 2. Process via RAG logic
        doc = Document(path=str(permanent_path), doc_metadata={"filename": filename})
        chunks = self.doc_service.ingest(doc)

        if not chunks:
            if permanent_path.exists():
                os.remove(permanent_path)
            return 0, 0 

        self.vector_store.add_chunks(chunks)
        return doc.id, len(chunks)

    def chat(self, message: str, conversation_id: int | None) -> dict:
        """
        Flow: User Query -> Vector Search -> History Context -> LLM Answer -> Save to DB
        """
        # 1. Manage Conversation
        if not conversation_id:
            conv = self.conv_manager.create_conversation(title=message[:30])
            conversation_id = conv.id

        if conversation_id is None:
            raise ValueError("Failed to create or retrieve conversation ID")

        # 2. Retrieve Context (Vector Search)
        # Search for top 5 relevant chunks
        search_results = self.vector_store.search(message, limit=5)
        context_str = "\n\n".join([res["text"] for res in search_results])

        # 3. Get Conversation History
        history = self.conv_manager.get_context(conversation_id)

        # 4. Generate Answer (RAG)
        response_text = self.llm.generate_with_context(
            query=message, context=context_str, conversation_history=history
        )

        # 5. Save Interaction
        self.conv_manager.add_message(conversation_id, "user", message)
        self.conv_manager.add_message(conversation_id, "assistant", response_text)

        return {
            "conversation_id": conversation_id,
            "response": response_text,
            "sources": [res["metadata"] for res in search_results],
        }
        
    def list_documents(self):
        statement = select(Document)
        docs = self.session.exec(statement).all()
        
        return [
            {
                "id": doc.id,
                "filename": doc.doc_metadata.get("filename", "Unknown"),
                "hash": doc.hash,
                "created_at": doc.doc_metadata.get("ingested_at")
            }
            for doc in docs
        ]
        
    def delete_document(self, doc_id: int):
        doc = self.session.get(Document, doc_id)
        
        if not doc:
            raise ValueError(f"Document with ID {doc_id} not found.")
        
        filename = doc.doc_metadata.get("filename")
        
        self.vector_store.delete_document(doc_id)
        self.session.delete(doc)
        self.session.commit()
        
        if filename:
            file_path = DOCUMENTS_DIR / filename
            if file_path.exists():
                os.remove(file_path)
                
        return filename