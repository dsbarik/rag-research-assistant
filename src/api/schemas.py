from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import datetime

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None

class ChatResponse(BaseModel):
    conversation_id: int
    response: str
    sources: List[Dict[str, Any]]

class IngestResponse(BaseModel):
    document_id: int
    chunks_processed: int
    status: str
    
class DocumentResponse(BaseModel):
    id: int
    filename: str
    created_at: datetime.datetime