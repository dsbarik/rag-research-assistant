from typing import List, Dict
from .schemas import Message


class ConversationContext:
    """Manages conversation context window"""
    
    def __init__(self, max_messages: int = 10):
        
        self.max_messages = max_messages
    
    def build_context(
        self,
        messages: List[Message],
        summary: str | None = None
    ) -> List[Dict[str, str]]:
        
        context = []
        
        # Add summary as system message if available
        if summary:
            context.append({
                "role": "system",
                "content": f"Previous conversation summary: {summary}"
            })
        
        # Get recent messages (within context window)
        recent_messages = messages[-self.max_messages:] if len(messages) > self.max_messages else messages
        
        # Convert to LLM format
        for msg in recent_messages:
            context.append({
                "role": msg.role,
                "content": msg.content
            })
        
        return context
    
    def should_summarize(self, message_count: int) -> bool:
        
        # Summarize when we have more than 2x the context window
        return message_count > (self.max_messages * 2)
    
    def estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        
        # Simple estimation: ~4 characters per token
        total_chars = sum(len(msg["content"]) for msg in messages)
        return total_chars // 4