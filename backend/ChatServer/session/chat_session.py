"""
Chat Session management for LLM conversations.

This module provides the ChatSession class that maintains the entire
conversation history, system prompts, and context for LLM interactions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from uuid import uuid4
import json


class MessageRole(str, Enum):
    """Role of the message sender."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"  # For function calling support
    TOOL = "tool"  # For tool use support


@dataclass
class Message:
    """
    Represents a single message in the conversation.
    
    Attributes:
        role: The role of the message sender
        content: The text content of the message
        timestamp: When the message was created
        message_id: Unique identifier for the message
        metadata: Additional metadata for the message
        name: Optional name for function/tool messages
        tool_call_id: Optional ID for tool responses
    """
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None
    tool_call_id: Optional[str] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary format."""
        result = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.name:
            result["name"] = self.name
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        return result
    
    def to_api_format(self) -> dict[str, str]:
        """Convert message to API-compatible format."""
        return self.to_dict()
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create a Message from a dictionary."""
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat())),
            message_id=data.get("message_id", str(uuid4())),
            metadata=data.get("metadata", {}),
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ChatSession:
    """
    Manages the entire chat session including conversation history and context.
    
    This class maintains the complete state of a conversation including:
    - System prompts
    - Message history
    - Session metadata
    - Context information
    
    The session can be serialized and passed to any LLM implementation
    for processing.
    
    Attributes:
        session_id: Unique identifier for the session
        system_prompt: The system prompt for the conversation
        messages: List of messages in the conversation
        created_at: When the session was created
        updated_at: When the session was last updated
        metadata: Additional session metadata
        max_history_length: Maximum number of messages to keep (None for unlimited)
        context: Additional context information for the session
    """
    session_id: str = field(default_factory=lambda: str(uuid4()))
    system_prompt: Optional[str] = None
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    max_history_length: Optional[int] = None
    context: dict[str, Any] = field(default_factory=dict)
    
    def add_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt.
        
        Args:
            prompt: The system prompt text
        """
        self.system_prompt = prompt
        self.updated_at = datetime.now()
    
    def add_message(
        self,
        role: MessageRole,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> Message:
        """
        Add a new message to the conversation.
        
        Args:
            role: The role of the message sender
            content: The text content of the message
            metadata: Optional metadata for the message
            name: Optional name for function/tool messages
            tool_call_id: Optional ID for tool responses
            
        Returns:
            The created Message object
        """
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            name=name,
            tool_call_id=tool_call_id,
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        
        # Trim history if needed
        if self.max_history_length and len(self.messages) > self.max_history_length:
            self.messages = self.messages[-self.max_history_length:]
        
        return message
    
    def add_user_message(self, content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
        """
        Add a user message to the conversation.
        
        Args:
            content: The user's message text
            metadata: Optional metadata
            
        Returns:
            The created Message object
        """
        return self.add_message(MessageRole.USER, content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[dict[str, Any]] = None) -> Message:
        """
        Add an assistant message to the conversation.
        
        Args:
            content: The assistant's message text
            metadata: Optional metadata
            
        Returns:
            The created Message object
        """
        return self.add_message(MessageRole.ASSISTANT, content, metadata)
    
    def get_messages(self, include_system: bool = True) -> list[Message]:
        """
        Get all messages in the conversation.
        
        Args:
            include_system: Whether to include the system prompt as a message
            
        Returns:
            List of messages
        """
        messages = []
        if include_system and self.system_prompt:
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content=self.system_prompt,
            ))
        messages.extend(self.messages)
        return messages
    
    def to_api_format(self) -> list[dict[str, str]]:
        """
        Convert the session to API-compatible format.
        
        Returns:
            List of message dictionaries for API consumption
        """
        messages = []
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })
        for msg in self.messages:
            messages.append(msg.to_api_format())
        return messages
    
    def clear_messages(self) -> None:
        """Clear all messages from the conversation (keeps system prompt)."""
        self.messages = []
        self.updated_at = datetime.now()
    
    def get_last_message(self) -> Optional[Message]:
        """
        Get the last message in the conversation.
        
        Returns:
            The last Message or None if no messages exist
        """
        return self.messages[-1] if self.messages else None
    
    def get_last_user_message(self) -> Optional[Message]:
        """
        Get the last user message in the conversation.
        
        Returns:
            The last user Message or None if no user messages exist
        """
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg
        return None
    
    def get_last_assistant_message(self) -> Optional[Message]:
        """
        Get the last assistant message in the conversation.
        
        Returns:
            The last assistant Message or None if no assistant messages exist
        """
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg
        return None
    
    def set_context(self, key: str, value: Any) -> None:
        """
        Set a context value.
        
        Args:
            key: The context key
            value: The context value
        """
        self.context[key] = value
        self.updated_at = datetime.now()
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """
        Get a context value.
        
        Args:
            key: The context key
            default: Default value if key doesn't exist
            
        Returns:
            The context value or default
        """
        return self.context.get(key, default)
    
    def get_message_count(self) -> int:
        """Get the total number of messages."""
        return len(self.messages)
    
    def get_token_estimate(self) -> int:
        """
        Get an estimated token count for the session.
        
        Returns:
            Estimated total tokens (rough estimate: 4 chars per token)
        """
        total_chars = 0
        if self.system_prompt:
            total_chars += len(self.system_prompt)
        for msg in self.messages:
            total_chars += len(msg.content)
        return total_chars // 4
    
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the session to a dictionary.
        
        Returns:
            Dictionary representation of the session
        """
        return {
            "session_id": self.session_id,
            "system_prompt": self.system_prompt,
            "messages": [
                {
                    "role": msg.role.value,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "message_id": msg.message_id,
                    "metadata": msg.metadata,
                    "name": msg.name,
                    "tool_call_id": msg.tool_call_id,
                }
                for msg in self.messages
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "max_history_length": self.max_history_length,
            "context": self.context,
        }
    
    def to_json(self) -> str:
        """
        Serialize the session to JSON.
        
        Returns:
            JSON string representation of the session
        """
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChatSession":
        """
        Create a ChatSession from a dictionary.
        
        Args:
            data: Dictionary representation of a session
            
        Returns:
            ChatSession instance
        """
        messages = [
            Message(
                role=MessageRole(msg["role"]),
                content=msg["content"],
                timestamp=datetime.fromisoformat(msg.get("timestamp", datetime.now().isoformat())),
                message_id=msg.get("message_id", str(uuid4())),
                metadata=msg.get("metadata", {}),
                name=msg.get("name"),
                tool_call_id=msg.get("tool_call_id"),
            )
            for msg in data.get("messages", [])
        ]
        
        return cls(
            session_id=data.get("session_id", str(uuid4())),
            system_prompt=data.get("system_prompt"),
            messages=messages,
            created_at=datetime.fromisoformat(data.get("created_at", datetime.now().isoformat())),
            updated_at=datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat())),
            metadata=data.get("metadata", {}),
            max_history_length=data.get("max_history_length"),
            context=data.get("context", {}),
        )
    
    @classmethod
    def from_json(cls, json_str: str) -> "ChatSession":
        """
        Create a ChatSession from a JSON string.
        
        Args:
            json_str: JSON string representation of a session
            
        Returns:
            ChatSession instance
        """
        return cls.from_dict(json.loads(json_str))
    
    def __repr__(self) -> str:
        return f"ChatSession(id={self.session_id[:8]}..., messages={len(self.messages)})"
