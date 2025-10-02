from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    
    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ChromaDB Configuration
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "support_docs"
    
    # Database Configuration
    database_url: str = "sqlite:///./data/conversations.db"
    
    # Redis Configuration (optional)
    redis_url: Optional[str] = None
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Logging
    log_level: str = "INFO"
    
    # Agent Configuration
    max_iterations: int = 10
    confidence_threshold: float = 0.7
    
    # RAG Configuration
    top_k_results: int = 5
    min_similarity_score: float = 0.5
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Create data directories
Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
Path("./data").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)