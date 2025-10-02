import streamlit as st
import requests
from datetime import datetime
import uuid
from typing import List, Dict
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import settings

# Page config
st.set_page_config(
    page_title="Support Agent Chat",
    page_icon="🤖",
    layout="wide"
)

# API endpoint
API_URL = f"http://{settings.api_host}:{settings.api_port}"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"


def send_message(message: str) -> Dict:
    """Send message to the API"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "message": message,
                "user_id": st.session_state.user_id,
                "session_id": st.session_state.session_id
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error communicating with agent: {e}")
        return None


def add_sample_documents():
    """Add sample documents to the knowledge base"""
    sample_docs = [
        {
            "content": "Our return policy allows returns within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days.",
            "metadata": {"source": "return_policy.pdf", "category": "policies"}
        },
        {
            "content": "We offer free shipping on orders over $50. Standard shipping takes 3-5 business days. Express shipping (1-2 days) is available for $15.",
            "metadata": {"source": "shipping_info.pdf", "category": "shipping"}
        },
        {
            "content": "Our customer support hours are Monday-Friday 9AM-6PM EST. You can reach us by phone at 1-800-SUPPORT or email at support@company.com.",
            "metadata": {"source": "contact_info.pdf", "category": "support"}
        },
        {
            "content": "Product warranty covers manufacturing defects for 1 year from purchase date. Extended warranty options are available at checkout.",
            "metadata": {"source": "warranty_info.pdf", "category": "policies"}
        },
        {
            "content": "To track your order, log into your account and go to Order History. You'll receive tracking numbers via email once your order ships.",
            "metadata": {"source": "order_tracking.pdf", "category": "orders"}
        }
    ]
    
    try:
        response = requests.post(
            f"{API_URL}/documents/bulk-add",
            json=sample_docs,
            timeout=30
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Error adding sample documents: {e}")
        return False


def get_kb_stats():
    """Get knowledge base statistics"""
    try:
        response = requests.get(f"{API_URL}/documents/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return None


# Sidebar
with st.sidebar:
    st.title("🤖 Support Agent")
    st.markdown("---")
    
    # Session info
    st.subheader("Session Info")
    st.text(f"User ID: {st.session_state.user_id[:12]}...")
    st.text(f"Session: {st.session_state.session_id[:12]}...")
    
    st.markdown("---")
    
    # Knowledge base management
    st.subheader("Knowledge Base")
    
    stats = get_kb_stats()
    if stats:
        st.metric("Documents", stats.get("count", 0))
    
    if st.button("Add Sample Documents"):
        with st.spinner("Adding sample documents..."):
            if add_sample_documents():
                st.success("Sample documents added!")
                st.rerun()
    
    st.markdown("---")
    
    # Controls
    st.subheader("Controls")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    if st.button("New Session"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"
        st.rerun()
    
    st.markdown("---")
    
    # Info
    st.subheader("About")
    st.info("""
    This is an intelligent support agent powered by:
    - **LangGraph** for agent orchestration
    - **RAG** for knowledge retrieval
    - **ChromaDB** for vector storage
    
    The agent can answer questions based on your knowledge base with citations and confidence scores.
    """)

# Main chat interface
st.title("💬 Customer Support Chat")
st.markdown("Ask me anything about our products, policies, or services!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata for assistant messages
        if message["role"] == "assistant" and "metadata" in message:
            metadata = message["metadata"]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                confidence = metadata.get("confidence", 0)
                st.caption(f"🎯 Confidence: {confidence:.0%}")
            
            with col2:
                intent = metadata.get("intent", "unknown")
                st.caption(f"🎭 Intent: {intent}")
            
            with col3:
                used_rag = metadata.get("used_rag", False)
                st.caption(f"📚 Used KB: {'✅' if used_rag else '❌'}")
            
            # Show sources if available
            if metadata.get("sources"):
                with st.expander("📖 Sources"):
                    for source in metadata["sources"]:
                        st.markdown(f"- {source}")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = send_message(prompt)
            
            if response_data:
                response_text = response_data.get("response", "Sorry, I couldn't process that.")
                confidence = response_data.get("confidence", 0)
                sources = response_data.get("sources", [])
                needs_clarification = response_data.get("needs_clarification", False)
                metadata_info = response_data.get("metadata", {})
                
                # Display response
                st.markdown(response_text)
                
                # Display metadata
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.caption(f"🎯 Confidence: {confidence:.0%}")
                
                with col2:
                    intent = metadata_info.get("intent", "unknown")
                    st.caption(f"🎭 Intent: {intent}")
                
                with col3:
                    used_rag = metadata_info.get("used_rag", False)
                    st.caption(f"📚 Used KB: {'✅' if used_rag else '❌'}")
                
                # Show sources
                if sources:
                    with st.expander("📖 Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "metadata": {
                        "confidence": confidence,
                        "sources": sources,
                        "intent": intent,
                        "used_rag": used_rag
                    }
                })
            else:
                error_msg = "Sorry, I encountered an error. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": {}
                })

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>Powered by LangGraph + ChromaDB | Built for scalable multi-channel support</small>
    </div>
    """,
    unsafe_allow_html=True
)