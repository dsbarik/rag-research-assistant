import shutil
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlmodel import Session
from typing import List

# Import your settings and infra
from config import DOCUMENTS_DIR
from infra.db.session import get_session
from orchestrator.service import OrchestratorService
from src.api.schemas import ChatRequest, ChatResponse, IngestResponse, DocumentResponse

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_session)
):
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="File must have a filename.")
    
    try:
        permanent_path = DOCUMENTS_DIR / file.filename
        
        with open(permanent_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        service = OrchestratorService(session)
        
        doc_id, chunks_processed = service.ingest_file(str(permanent_path), file.filename)
        
        return IngestResponse(**{
            "status": "success",
            "chunks_processed": chunks_processed,
            "document_id": doc_id  
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    session: Session = Depends(get_session)
):
    try:
        service = OrchestratorService(session)
        result = service.chat(request.message, request.conversation_id)
        return ChatResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.get("/documents", response_model=List[DocumentResponse])
async def list_documents(session: Session = Depends(get_session)):
    try:
        service = OrchestratorService(session)
        return service.list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: int, session: Session = Depends(get_session)):
    service = OrchestratorService(session)
    
    try:
        filename = service.delete_document(doc_id)
        return {
            "status": "success",
            "filename": filename,
            "document_id": doc_id
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))