from pathlib import Path
from typing import Tuple, Dict, Any

import fitz


def load_text(path: str) -> Tuple[str, Dict[str, Any]]:
    
    p = Path(path)
    
    if not p.exists():
        raise FileNotFoundError(path)
    
    suffix = p.suffix.lower()
    
    if suffix == ".pdf":
        doc = fitz.open(p)
        
        pages_text = []
        
        for page in doc:
            text = page.get_text("text")
            if text:
                pages_text.append(text)
                
        full_text = "\n".join(pages_text)
        
        metadata = {
            "file_type": "pdf",
            "pages": doc.page_count
        }
        
        return full_text, metadata
    
    if suffix in (".txt", ".md"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        return text, {"file_type": "txt"}
    
    raise ValueError(f"Unsupported file type: {suffix}")