import datetime
import logging
from typing import Any, Dict, List, Optional

from sqlmodel import Session, desc, select

from .schemas import Conversation, Message

logger = logging.getLogger(__name__)


class ConversationManager:
    """Manages conversation lifecycle and message history."""

    def __init__(self, session: Session, max_context_messages: int = 10) -> None:
        self.session = session
        self.max_context_messages = max_context_messages

    # ------------------------------------------------------------------ #
    #  Conversation CRUD                                                   #
    # ------------------------------------------------------------------ #

    def create_conversation(self, title: str = "New Conversation") -> Conversation:
        conversation = Conversation(title=title)
        self.session.add(conversation)
        self.session.commit()
        self.session.refresh(conversation)
        return conversation

    def get_conversation(self, conversation_id: int) -> Optional[Conversation]:
        return self.session.get(Conversation, conversation_id)

    def delete_conversation(self, conversation_id: int) -> bool:
        """
        Delete a conversation and all its messages (cascade).
        Returns True if deleted, False if not found.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        self.session.delete(conversation)
        self.session.commit()
        return True

    def list_conversations(self, limit: int = 50) -> List[Conversation]:
        statement = (
            select(Conversation)
            .order_by(desc(Conversation.updated_at))
            .limit(limit)
        )
        return list(self.session.exec(statement).all())

    # ------------------------------------------------------------------ #
    #  Messages                                                            #
    # ------------------------------------------------------------------ #

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            mes_metadata=metadata or {},
        )
        self.session.add(message)

        # Bump conversation updated_at
        conversation = self.get_conversation(conversation_id)
        if conversation:
            conversation.updated_at = datetime.datetime.now(datetime.timezone.utc)

        self.session.commit()
        self.session.refresh(message)
        return message

    def get_messages(self, conversation_id: int) -> List[Message]:
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.timestamp)  # type: ignore[arg-type]
        )
        return list(self.session.exec(statement).all())

    # ------------------------------------------------------------------ #
    #  Context window                                                      #
    # ------------------------------------------------------------------ #

    def get_context(self, conversation_id: int) -> List[Dict[str, str]]:
        """
        Return the most recent messages as a list of role/content dicts
        suitable for passing directly to the LLM.

        Applies a sliding window of `max_context_messages` to keep the
        context size bounded.
        """
        messages = self.get_messages(conversation_id)
        recent = (
            messages[-self.max_context_messages :]
            if len(messages) > self.max_context_messages
            else messages
        )
        return [{"role": msg.role, "content": msg.content} for msg in recent]