"""Utility functions for Legal Document Analyzer"""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document

def ensure_data_directory():
    """Ensure data directory exists"""
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    return data_dir

def ensure_database_directory():
    """Ensure database directory exists"""
    db_dir = Path("./database")
    db_dir.mkdir(exist_ok=True)
    return db_dir

def get_pdf_files():
    """Get list of PDF files in data directory"""
    data_dir = ensure_data_directory()
    return list(data_dir.glob("*.pdf"))

def extract_document_info(documents: List[Document]) -> dict:
    """Extract information from documents"""
    info = {
        "total_pages": len(documents),
        "total_chars": sum(len(doc.page_content) for doc in documents),
        "sources": list(set(doc.metadata.get("source_file", "Unknown") for doc in documents))
    }
    return info

def format_document_for_display(content: str, max_length: int = 500) -> str:
    """Format document content for display"""
    if len(content) > max_length:
        return content[:max_length] + "..."
    return content

def save_summary_to_file(summary: str, filename: str) -> str:
    """Save summary to file"""
    output_dir = Path("./summaries")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        f.write(summary)
    
    return str(filepath)

def clean_database():
    """Clean the Chroma database"""
    db_dir = Path("./database")
    if db_dir.exists():
        import shutil
        shutil.rmtree(db_dir)
        db_dir.mkdir(exist_ok=True)
        return True
    return False

def get_database_stats() -> dict:
    """Get statistics about the database"""
    db_dir = Path("./database")
    
    if not db_dir.exists():
        return {"exists": False, "size": 0, "files": 0}
    
    total_size = sum(f.stat().st_size for f in db_dir.rglob("*") if f.is_file())
    files_count = len(list(db_dir.rglob("*")))
    
    return {
        "exists": True,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "files": files_count
    }
