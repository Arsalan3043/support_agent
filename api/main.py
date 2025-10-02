from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime
from loguru import logger

from core.config import settings
from agents.support_agent import support_agent
from channels.web_adapter import web_adapter
from storage.vector_store import vector_store

# Configure logging
logger.add("logs/api.log", rotation="500 MB", retention="10 days", level=settings.log_level)

app = FastAPI(
    title="Support Agent API",
    description="Agentic RAG-based customer support system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "default_user"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]
    needs_clarification: bool
    session_id: str
    metadata: Dict[str, Any]


class DocumentRequest(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
    doc_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    vector_store_count: int


# Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Support Agent API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        stats = vector_store.get_collection_stats()
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            vector_store_count=stats.get("count", 0)
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for interacting with the support agent
    """
    try:
        logger.info(f"Chat request from user {request.user_id}")
        
        # Convert to standard message using adapter
        std_message = web_adapter.receive_message(
            user_id=request.user_id,
            content=request.message,
            session_id=request.session_id
        )
        
        # Process with agent
        agent_response = support_agent.process_message(
            user_message=request.message,
            session_id=std_message.session_id
        )
        
        # Send response through adapter
        web_adapter.send_message(agent_response, request.user_id)
        
        return ChatResponse(
            response=agent_response.content,
            confidence=agent_response.confidence,
            sources=agent_response.sources,
            needs_clarification=agent_response.needs_clarification,
            session_id=std_message.session_id,
            metadata=agent_response.metadata
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.post("/documents/add")
async def add_document(request: DocumentRequest):
    """
    Add a document to the knowledge base
    """
    try:
        logger.info("Adding document to knowledge base")
        
        metadata = request.metadata or {}
        metadata["added_at"] = datetime.now().isoformat()
        
        vector_store.add_documents(
            documents=[request.content],
            metadatas=[metadata],
            ids=[request.doc_id] if request.doc_id else None
        )
        
        return {
            "status": "success",
            "message": "Document added successfully",
            "doc_id": request.doc_id
        }
    
    except Exception as e:
        logger.error(f"Error adding document: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding document: {str(e)}")


@app.post("/documents/bulk-add")
async def bulk_add_documents(documents: List[DocumentRequest]):
    """
    Add multiple documents to the knowledge base
    """
    try:
        logger.info(f"Adding {len(documents)} documents to knowledge base")
        
        contents = [doc.content for doc in documents]
        metadatas = []
        ids = []
        
        for doc in documents:
            metadata = doc.metadata or {}
            metadata["added_at"] = datetime.now().isoformat()
            metadatas.append(metadata)
            ids.append(doc.doc_id if doc.doc_id else None)
        
        vector_store.add_documents(
            documents=contents,
            metadatas=metadatas,
            ids=ids if any(ids) else None
        )
        
        return {
            "status": "success",
            "message": f"Added {len(documents)} documents successfully"
        }
    
    except Exception as e:
        logger.error(f"Error bulk adding documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error bulk adding documents: {str(e)}")


@app.get("/documents/stats")
async def get_document_stats():
    """
    Get statistics about the knowledge base
    """
    try:
        stats = vector_store.get_collection_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@app.delete("/documents/collection")
async def delete_collection():
    """
    Delete the entire knowledge base collection (use with caution!)
    """
    try:
        vector_store.delete_collection()
        return {
            "status": "success",
            "message": "Collection deleted successfully"
        }
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


@app.post("/webhook/instagram")
async def instagram_webhook(payload: Dict[str, Any]):
    """
    Webhook endpoint for Instagram (placeholder for future)
    """
    logger.info("Instagram webhook received")
    return {
        "status": "success",
        "message": "Instagram integration coming soon"
    }


@app.post("/webhook/whatsapp")
async def whatsapp_webhook(payload: Dict[str, Any]):
    """
    Webhook endpoint for WhatsApp (placeholder for future)
    """
    logger.info("WhatsApp webhook received")
    return {
        "status": "success",
        "message": "WhatsApp integration coming soon"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )