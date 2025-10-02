from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ChannelType(str, Enum):
    """Supported communication channels"""
    WEB = "web"
    CLI = "cli"
    INSTAGRAM = "instagram"
    WHATSAPP = "whatsapp"
    SLACK = "slack"


class MessageType(str, Enum):
    """Types of messages"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    AUDIO = "audio"
    VIDEO = "video"


class StandardMessage(BaseModel):
    """Unified message format across all channels"""
    message_id: str = Field(default_factory=lambda: f"msg_{datetime.now().timestamp()}")
    channel: ChannelType
    user_id: str
    session_id: str
    content: str
    message_type: MessageType = MessageType.TEXT
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AgentResponse(BaseModel):
    """Response from an agent"""
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    reasoning: Optional[str] = None


class ConversationState(BaseModel):
    """State maintained throughout a conversation"""
    session_id: str
    user_id: str
    channel: ChannelType
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    current_intent: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class RAGResult(BaseModel):
    """Result from RAG retrieval"""
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str


class ToolResult(BaseModel):
    """Result from tool execution"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    confidence: float = 1.0