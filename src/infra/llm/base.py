from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseLLM(ABC):
    
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        pass
    
    @abstractmethod
    def generate_with_context(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] | None = None,
        temperature: float = 0.7
    ) -> str:
        pass