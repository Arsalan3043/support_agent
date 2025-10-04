import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from loguru import logger
import uuid

from core.config import settings
from core.models import RAGResult


class VectorStore:
    """Wrapper for ChromaDB vector store operations"""
    
    def __init__(self):
        self.persist_directory = settings.chroma_persist_dir
        self.collection_name = settings.chroma_collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """Add documents to the vector store"""
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in documents]
            
            if metadatas is None:
                metadatas = [{} for _ in documents]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(documents)
            
            # Add to collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RAGResult]:
        """Search for similar documents"""
        try:
            if k is None:
                k = settings.top_k_results
            
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter
            )
            
            # Convert to RAGResult objects
            rag_results = []
            for i in range(len(results['documents'][0])):
                score = 1 - results['distances'][0][i]  # Convert distance to similarity
                
                if score >= settings.min_similarity_score:
                    rag_results.append(RAGResult(
                        content=results['documents'][0][i],
                        score=score,
                        metadata=results['metadatas'][0][i],
                        source=results['metadatas'][0][i].get('source', 'unknown')
                    ))
            
            logger.info(f"Found {len(rag_results)} relevant documents for query")
            return rag_results
        
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def delete_collection(self) -> None:
        """Delete the entire collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_model": settings.embedding_model
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}


# Global vector store instance
vector_store = VectorStore()