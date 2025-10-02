from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from core.models import StandardMessage, AgentResponse


class ChannelAdapter(ABC):
    """Base class for all channel adapters"""
    
    def __init__(self, channel_name: str):
        self.channel_name = channel_name
    
    @abstractmethod
    def receive_message(self) -> StandardMessage:
        """Receive a message from the channel and convert to StandardMessage"""
        pass
    
    @abstractmethod
    def send_message(self, message: AgentResponse, recipient_id: str) -> bool:
        """Send a message through the channel"""
        pass
    
    @abstractmethod
    def handle_media(self, media_data: Any) -> Dict[str, Any]:
        """Handle media attachments (images, files, etc.)"""
        pass
    
    def validate_message(self, message: StandardMessage) -> bool:
        """Validate message format"""
        return bool(message.content and message.user_id and message.session_id)
    
    def format_response(self, response: AgentResponse) -> str:
        """Format agent response for the channel"""
        formatted = response.content
        
        # Add sources if available
        if response.sources:
            formatted += "\n\n📚 Sources:\n"
            for source in response.sources[:3]:  # Limit to 3 sources
                formatted += f"• {source}\n"
        
        return formatted