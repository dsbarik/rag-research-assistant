from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlmodel import Session
from typing import List

from src.infra.db.session import get_session
from src.orchestrator.service import OrchestratorService
from src.api.schemas import ChatRequest, ChatResponse, IngestResponse, DocumentResponse
from src.documents.service import DocumentService
from src.conversation.manager import ConversationManager
from src.retrieval.service import RetrievalService
from src.infra.llm import OllamaLLM
from src.config.settings import LLM_MODEL_NAME
from src.utils.exceptions import ServiceException, ResourceNotFoundException, InvalidRequestException

router = APIRouter()

def get_orchestrator(session: Session = Depends(get_session)) -> OrchestratorService:
    doc_service = DocumentService(session)
    conv_manager = ConversationManager(session)
    llm = OllamaLLM(model=LLM_MODEL_NAME)
    retrieval_service = RetrievalService(llm)
    return OrchestratorService(session, doc_service, conv_manager, retrieval_service, llm)

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename.")

    try:
        result = orchestrator.ingest_file(file.file, file.filename)
        return IngestResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    orchestrator: OrchestratorService = Depends(get_orchestrator)
):
    try:
        result = orchestrator.chat(request.message, request.conversation_id)
        return ChatResponse(**result)
    except InvalidRequestException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(orchestrator: OrchestratorService = Depends(get_orchestrator)):
    try:
        return orchestrator.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, orchestrator: OrchestratorService = Depends(get_orchestrator)):
    try:
        filename = orchestrator.delete_document(doc_id)
        return {
            "status": "success",
            "filename": filename,
            "document_id": doc_id
        }
    except ResourceNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))