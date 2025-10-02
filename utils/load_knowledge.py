"""
Utility script to load knowledge base documents into ChromaDB
"""
import sys
from pathlib import Path
import json
from typing import List, Dict
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import your vector_store
from storage.vector_store import vector_store


def load_sample_knowledge_base():
    """Load sample knowledge base for testing"""
    
    sample_documents = [
        {
            "content": """
            Return Policy: We accept returns within 30 days of purchase. 
            To be eligible for a return, items must be unused, in their original packaging, 
            and include all tags and accessories. Refunds are processed within 5-7 business days 
            after we receive your return. Original shipping costs are non-refundable.
            """,
            "metadata": {
                "source": "return_policy.pdf",
                "category": "policy",
                "department": "customer_service"
            }
        },
        {
            "content": """
            Shipping Information: We offer free standard shipping on all orders over $50. 
            Standard shipping typically takes 3-5 business days. Express shipping (1-2 business days) 
            is available for $15. International shipping varies by location and typically takes 7-14 business days.
            All orders are shipped Monday through Friday, excluding holidays.
            """,
            "metadata": {
                "source": "shipping_policy.pdf",
                "category": "shipping",
                "department": "logistics"
            }
        },
        # Add the rest of your sample_documents here...
    ]
    
    logger.info(f"Loading {len(sample_documents)} sample documents...")
    
    contents = [doc["content"].strip() for doc in sample_documents]
    metadatas = [doc["metadata"] for doc in sample_documents]
    
    vector_store.add_documents(documents=contents, metadatas=metadatas)
    
    logger.info(f"Successfully loaded {len(sample_documents)} documents into the knowledge base")
    
    stats = vector_store.get_collection_stats()
    logger.info(f"Collection stats: {stats}")


def load_from_json(file_path: str):
    """Load documents from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error("JSON file must contain a list of documents")
            return
        
        contents = [doc.get("content", "") for doc in data]
        metadatas = [doc.get("metadata", {}) for doc in data]
        
        vector_store.add_documents(documents=contents, metadatas=metadatas)
        logger.info(f"Successfully loaded {len(contents)} documents from {file_path}")
    
    except Exception as e:
        logger.error(f"Error loading JSON file: {e}")


def load_from_text_files(directory: str):
    """Load documents from text files in a directory"""
    try:
        path = Path(directory)
        text_files = list(path.glob("*.txt")) + list(path.glob("*.md"))
        
        if not text_files:
            logger.warning(f"No text files found in {directory}")
            return
        
        contents = []
        metadatas = []
        
        for file in text_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    content = f.read()
                contents.append(content)
                metadatas.append({
                    "source": file.name,
                    "file_path": str(file)
                })
            except Exception as fe:
                logger.error(f"Failed to read {file}: {fe}")
        
        if contents:
            vector_store.add_documents(documents=contents, metadatas=metadatas)
            logger.info(f"Successfully loaded {len(contents)} documents from {directory}")
    
    except Exception as e:
        logger.error(f"Error loading from directory {directory}: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load documents into the knowledge base")
    parser.add_argument("--sample", action="store_true", help="Load sample documents")
    parser.add_argument("--json", type=str, help="Path to JSON file containing documents")
    parser.add_argument("--directory", type=str, help="Directory containing text files to load")
    args = parser.parse_args()
    
    if args.sample:
        load_sample_knowledge_base()
    elif args.json:
        load_from_json(args.json)
    elif args.directory:
        load_from_text_files(args.directory)
    else:
        print("Please specify --sample, --json, or --directory")
        parser.print_help()
        logger.error("No source specified. Use --sample, --json, or --directory")
