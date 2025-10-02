from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from pydantic import Field
from loguru import logger

from storage.vector_store import vector_store
from core.models import ToolResult


class RAGTool(BaseTool):
    """Tool for retrieving information from the knowledge base using RAG"""
    
    name: str = "knowledge_base_search"
    description: str = """
    Search the knowledge base for relevant information to answer customer questions.
    Use this tool when you need to find specific information about products, policies, 
    procedures, or any documented information.
    
    Input should be a clear, specific question or search query.
    The tool returns relevant documents with confidence scores and sources.
    """
    
    return_direct: bool = False
    
    def _run(self, query: str) -> str:
        """Execute the RAG search"""
        try:
            logger.info(f"RAG Tool: Searching for: {query}")
            
            # Perform similarity search
            results = vector_store.similarity_search(query)
            
            if not results:
                return self._format_no_results()
            
            # Format results
            return self._format_results(results, query)
        
        except Exception as e:
            logger.error(f"Error in RAG tool: {e}")
            return f"Error searching knowledge base: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version"""
        return self._run(query)
    
    def _format_results(self, results: List, query: str) -> str:
        """Format search results for the agent"""
        if not results:
            return self._format_no_results()
        
        # Calculate average confidence
        avg_confidence = sum(r.score for r in results) / len(results)
        
        formatted = f"Found {len(results)} relevant documents (avg confidence: {avg_confidence:.2f}):\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"[Document {i}] (confidence: {result.score:.2f})\n"
            formatted += f"Source: {result.source}\n"
            formatted += f"Content: {result.content}\n"
            
            # Add metadata if available
            if result.metadata:
                metadata_str = ", ".join([f"{k}: {v}" for k, v in result.metadata.items() if k != 'source'])
                if metadata_str:
                    formatted += f"Metadata: {metadata_str}\n"
            
            formatted += "\n"
        
        # Add usage instructions
        if avg_confidence < 0.7:
            formatted += "\n⚠️ Note: Confidence is moderate. Consider asking clarifying questions.\n"
        
        return formatted
    
    def _format_no_results(self) -> str:
        """Format response when no results found"""
        return """
No relevant information found in the knowledge base.
This might mean:
1. The information is not in our documentation
2. The query needs to be rephrased
3. This requires human assistance

Consider asking the user for clarification or escalating to a human agent.
"""


class RAGToolWithMetadata:
    """Enhanced RAG tool that returns structured results"""
    
    def __init__(self):
        self.vector_store = vector_store
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> ToolResult:
        """
        Search with structured output including confidence and metadata
        """
        try:
            results = self.vector_store.similarity_search(query, k=k, filter=filter)
            
            if not results:
                return ToolResult(
                    tool_name="rag_search",
                    success=True,
                    result=[],
                    confidence=0.0
                )
            
            # Calculate confidence
            avg_confidence = sum(r.score for r in results) / len(results)
            
            return ToolResult(
                tool_name="rag_search",
                success=True,
                result=[{
                    "content": r.content,
                    "score": r.score,
                    "source": r.source,
                    "metadata": r.metadata
                } for r in results],
                confidence=avg_confidence
            )
        
        except Exception as e:
            logger.error(f"RAG search error: {e}")
            return ToolResult(
                tool_name="rag_search",
                success=False,
                result=None,
                error=str(e),
                confidence=0.0
            )


# Create tool instances
rag_tool = RAGTool()
rag_tool_with_metadata = RAGToolWithMetadata()