from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from channels.base import ChannelAdapter
from core.models import StandardMessage, AgentResponse, ChannelType, MessageType
from loguru import logger


class WebAdapter(ChannelAdapter):
    """Adapter for web-based chat interface"""
    
    def __init__(self):
        super().__init__("web")
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def receive_message(self, user_id: str, content: str, session_id: Optional[str] = None) -> StandardMessage:
        """Receive message from web interface"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Initialize session if new
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "user_id": user_id,
                "created_at": datetime.now(),
                "message_count": 0
            }
        
        self.sessions[session_id]["message_count"] += 1
        
        message = StandardMessage(
            message_id=f"web_{uuid.uuid4()}",
            channel=ChannelType.WEB,
            user_id=user_id,
            session_id=session_id,
            content=content,
            message_type=MessageType.TEXT,
            metadata={
                "session_message_count": self.sessions[session_id]["message_count"]
            }
        )
        
        logger.info(f"Web adapter received message from user {user_id}")
        return message
    
    def send_message(self, message: AgentResponse, recipient_id: str) -> bool:
        """Send message through web interface"""
        try:
            # In a real implementation, this would push to websocket or queue
            formatted = self.format_response(message)
            logger.info(f"Web adapter sending message to {recipient_id}")
            return True
        except Exception as e:
            logger.error(f"Error sending web message: {e}")
            return False
    
    def handle_media(self, media_data: Any) -> Dict[str, Any]:
        """Handle media uploads from web"""
        return {
            "supported": True,
            "type": "file",
            "url": None  # Would process and return URL
        }
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information"""
        return self.sessions.get(session_id)


# Global instance
web_adapter = WebAdapter()