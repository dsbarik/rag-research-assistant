from typing import List, Dict, Optional
import requests
from .base import BaseLLM

class OllamaLLM(BaseLLM):
    
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434"
    ):
        self.model = model
        self.base_url = base_url
        self.generate_endpoint = f"{base_url}/api/generate"
        self.tags_endpoint = f"{base_url}/api/tags"
    
    def is_available(self) -> bool:
        try:
            response = requests.get(self.tags_endpoint, timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def list_models(self) -> List[str]:
        try:
            response = requests.get(self.tags_endpoint)
            response.raise_for_status()
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        except requests.RequestException:
            return []
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        
        prompt = self._format_messages(messages)
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        if max_tokens:
            payload["options"]["num_predict"] = max_tokens
        
        response = requests.post(self.generate_endpoint, json=payload)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "")
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        return "\n\n".join(formatted)
    
    def generate_with_context(
        self,
        query: str,
        context: str,
        conversation_history: List[Dict[str, str]] | None = None,
        temperature: float = 0.7
    ) -> str:
        
        messages = []
        
        # 1. System Prompt with Context
        system_content = (
            "You are a helpful research assistant. "
            "Use the provided context to answer the user's question.\n\n"
            f"Context:\n{context}"
        )
        
        messages.append({"role": "system", "content": system_content})
        
        # 2. Add History
        if conversation_history:
            messages.extend(conversation_history)
        
        # 3. Add User Query
        messages.append({"role": "user", "content": query})
        
        return self.generate(messages, temperature=temperature)