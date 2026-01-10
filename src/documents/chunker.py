from typing import List
import uuid


from .schemas import Document


def chunk_text(
    text: str,
    document: Document,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[dict]:
    
    if not text:
        return []
    
    chunks: List[dict] = []
    start = 0
    chunk_index = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text = text[start:end]
        
        chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "document_id": str(document.id),
            "text": chunk_text,
            "metadata": {
                "chunk_index": chunk_index,
                "char_start": start,
                "char_end": end,
                **document.doc_metadata
            }
        })
        
        chunk_index += 1

        if end == text_length:
            break
        
        start = max(start + 1, end - overlap)
        
    return chunks
    