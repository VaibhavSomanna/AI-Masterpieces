from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Qdrant settings
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "rag_documents"
    
    # Model settings
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # OpenAI settings (if using OpenAI for generation)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    
    # Hugging Face settings (if using HF models)
    hf_model_name: str = "microsoft/DialoGPT-medium"
    
    # Document processing settings
    max_chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Upload settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = [".pdf", ".docx", ".txt", ".png", ".jpg", ".jpeg"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"