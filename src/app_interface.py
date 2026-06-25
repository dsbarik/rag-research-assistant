from sqlmodel import SQLModel

from src.infra.db.session import get_session, engine
from src.orchestrator.service import OrchestratorService
from src.documents.service import DocumentService
from src.conversation.manager import ConversationManager
from src.retrieval.service import RetrievalService
from src.infra.llm import OllamaLLM
from src.config.settings import LLM_MODEL_NAME


class ResearchAssistant:
    def __init__(self) -> None:
        SQLModel.metadata.create_all(engine)
        

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            with get_session() as session:
                doc_service = DocumentService(session)
                conv_manager = ConversationManager(session)
                llm = OllamaLLM(model=LLM_MODEL_NAME)
                retrieval_service = RetrievalService(llm)
                
                service = OrchestratorService(
                    session=session,
                    doc_service=doc_service,
                    conv_manager=conv_manager,
                    retrieval_service=retrieval_service,
                    llm=llm
                )

                method = getattr(service, name)

                return method(*args, **kwargs)

        return wrapper
