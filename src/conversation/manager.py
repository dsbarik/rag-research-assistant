from typing import Optional, List, Dict, Any
import datetime
from sqlmodel import Session, select, desc
from .schemas import Conversation, Message, ConversationSummary
from .context import ConversationContext


class ConversationManager:
    """Manages conversation lifecycle and context"""
    
    def __init__(self, session: Session, max_context_messages: int = 10):
        
        self.session = session
        self.context_manager = ConversationContext(max_messages=max_context_messages)
    
    def create_conversation(self, title: str = "New Conversation") -> Conversation:
        
        conversation = Conversation(title=title)
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation
    
    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        
        return self.session.get(Conversation, conversation_id)
    
    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Dict[str, Any] | None = None
    ) -> Message:
        
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            mes_metadata=metadata or {}
        )
        self.session.add(message)
        
        # Update conversation timestamp
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.updated_at = datetime.datetime.now(datetime.timezone.utc)
        
        self.session.commit()
        self.session.refresh(message)
        return message
    
    def get_messages(self, conversation_id: int) -> List[Message]:
        
        statement = select(Message).where(
            Message.conversation_id == conversation_id
        ).order_by(Message.timestamp) # type: ignore
        
        return list(self.session.exec(statement).all())
    
    def get_context(self, conversation_id: int) -> List[Dict[str, str]]:
        
        messages = self.get_messages(conversation_id)
        
        # Check if we have a summary
        summary = self.get_summary(conversation_id)
        summary_text = summary.summary if summary else None
        
        return self.context_manager.build_context(messages, summary_text)
    
    def get_summary(self, conversation_id: int) -> Optional[ConversationSummary]:
        
        statement = select(ConversationSummary).where(
            ConversationSummary.conversation_id == conversation_id
        )
        return self.session.exec(statement).first()
    
    def create_summary(self, conversation_id: int, summary_text: str) -> ConversationSummary:
        
        existing = self.get_summary(conversation_id)
        messages = self.get_messages(conversation_id)
        
        if existing:
            existing.summary = summary_text
            existing.messages_summarized = len(messages)
            existing.updated_at = datetime.datetime.now(datetime.timezone.utc)
            self.session.commit()
            self.session.refresh(existing)
            return existing
        else:
            summary = ConversationSummary(
                conversation_id=conversation_id,
                summary=summary_text,
                messages_summarized=len(messages)
            )
            self.session.add(summary)
            self.session.commit()
            self.session.refresh(summary)
            return summary
    
    def should_summarize(self, conversation_id: int) -> bool:
        
        messages = self.get_messages(conversation_id)
        return self.context_manager.should_summarize(len(messages))
    
    def delete_conversation(self, conversation_id: int) -> bool:
        
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        self.session.delete(conversation)
        self.session.commit()
        return True
    
    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        
        statement = select(Conversation).order_by(
            desc(Conversation.updated_at)
        ).limit(limit)
        
        return list(self.session.exec(statement).all())