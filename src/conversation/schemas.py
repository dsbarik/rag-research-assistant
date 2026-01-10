from typing import Optional, List, Dict, Any
import datetime
from sqlmodel import SQLModel, Field, Relationship
from sqlalchemy import Column, JSON


class Conversation(SQLModel, table=True):
    """Represents a conversation session"""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    title: Optional[str] = Field(default="New Conversation", max_length=255)
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    conv_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Relationship to messages
    messages: List["Message"] = Relationship(back_populates="conversation")


class Message(SQLModel, table=True):
    """Represents a single message in a conversation"""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True)
    role: str = Field(max_length=50)  # "user", "assistant", "system"
    content: str
    timestamp: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    mes_metadata: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    
    # Relationship to conversation
    conversation: Optional[Conversation] = Relationship(back_populates="messages")


class ConversationSummary(SQLModel, table=True):
    """Stores summarized conversation history for memory management"""
    
    id: Optional[int] = Field(default=None, primary_key=True)
    conversation_id: int = Field(foreign_key="conversation.id", index=True, unique=True)
    summary: str  # Condensed summary of conversation history
    messages_summarized: int = Field(default=0)  # Number of messages included in summary
    created_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    updated_at: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))