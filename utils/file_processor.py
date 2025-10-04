"""
File processor for handling different document formats
"""
import io
from typing import List, Dict, Any, Tuple
from pathlib import Path
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from loguru import logger


class FileProcessor:
    """Process different file formats and extract text content"""
    
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.pdf', '.txt', '.md', '.docx', '.doc'}
    
    @staticmethod
    def process_file(file_content: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Process uploaded file and return documents and metadata
        
        Args:
            file_content: Raw file bytes
            filename: Original filename with extension
            
        Returns:
            Tuple of (documents list, metadata list)
        """
        extension = Path(filename).suffix.lower()
        
        if extension not in FileProcessor.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {extension}. Supported: {FileProcessor.SUPPORTED_EXTENSIONS}")
        
        logger.info(f"Processing file: {filename} (type: {extension})")
        
        # Route to appropriate processor
        if extension == '.csv':
            return FileProcessor._process_csv(file_content, filename)
        elif extension in ['.xlsx', '.xls']:
            return FileProcessor._process_excel(file_content, filename)
        elif extension == '.pdf':
            return FileProcessor._process_pdf(file_content, filename)
        elif extension in ['.txt', '.md']:
            return FileProcessor._process_text(file_content, filename)
        elif extension in ['.docx', '.doc']:
            return FileProcessor._process_docx(file_content, filename)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
    
    @staticmethod
    def _process_csv(file_content: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process CSV file"""
        try:
            # Read CSV with pandas
            df = pd.read_csv(io.BytesIO(file_content))
            
            logger.info(f"CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            
            documents = []
            metadatas = []
            
            # Option 1: Each row becomes a document
            for idx, row in df.iterrows():
                # Create document from row
                doc_text = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                
                documents.append(doc_text)
                metadatas.append({
                    "source": filename,
                    "row_number": int(idx),
                    "file_type": "csv"
                })
            
            logger.info(f"Extracted {len(documents)} documents from CSV")
            return documents, metadatas
        
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            raise ValueError(f"Failed to process CSV file: {str(e)}")
    
    @staticmethod
    def _process_excel(file_content: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process Excel file"""
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(io.BytesIO(file_content))
            
            documents = []
            metadatas = []
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                logger.info(f"Sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
                
                # Each row becomes a document
                for idx, row in df.iterrows():
                    doc_text = "\n".join([f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col])])
                    
                    documents.append(doc_text)
                    metadatas.append({
                        "source": filename,
                        "sheet_name": sheet_name,
                        "row_number": int(idx),
                        "file_type": "excel"
                    })
            
            logger.info(f"Extracted {len(documents)} documents from Excel")
            return documents, metadatas
        
        except Exception as e:
            logger.error(f"Error processing Excel: {e}")
            raise ValueError(f"Failed to process Excel file: {str(e)}")
    
    @staticmethod
    def _process_pdf(file_content: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process PDF file"""
        try:
            pdf_reader = PdfReader(io.BytesIO(file_content))
            
            documents = []
            metadatas = []
            
            # Extract text from each page
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                
                if text.strip():  # Only add non-empty pages
                    documents.append(text)
                    metadatas.append({
                        "source": filename,
                        "page_number": page_num + 1,
                        "file_type": "pdf",
                        "total_pages": len(pdf_reader.pages)
                    })
            
            logger.info(f"Extracted {len(documents)} pages from PDF")
            return documents, metadatas
        
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise ValueError(f"Failed to process PDF file: {str(e)}")
    
    @staticmethod
    def _process_text(file_content: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process text/markdown file"""
        try:
            text = file_content.decode('utf-8')
            
            # Split into chunks if text is very long (optional)
            # For now, treat entire file as one document
            documents = [text]
            metadatas = [{
                "source": filename,
                "file_type": "text",
                "char_count": len(text)
            }]
            
            logger.info(f"Extracted text document ({len(text)} chars)")
            return documents, metadatas
        
        except Exception as e:
            logger.error(f"Error processing text file: {e}")
            raise ValueError(f"Failed to process text file: {str(e)}")
    
    @staticmethod
    def _process_docx(file_content: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Process Word document"""
        try:
            doc = Document(io.BytesIO(file_content))
            
            # Extract paragraphs
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Combine all paragraphs
            full_text = "\n\n".join(paragraphs)
            
            documents = [full_text]
            metadatas = [{
                "source": filename,
                "file_type": "docx",
                "paragraph_count": len(paragraphs)
            }]
            
            logger.info(f"Extracted Word document ({len(paragraphs)} paragraphs)")
            return documents, metadatas
        
        except Exception as e:
            logger.error(f"Error processing Word document: {e}")
            raise ValueError(f"Failed to process Word document: {str(e)}")
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Return list of supported file extensions"""
        return list(FileProcessor.SUPPORTED_EXTENSIONS)


# Global instance
file_processor = FileProcessor()