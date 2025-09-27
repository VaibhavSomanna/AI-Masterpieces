from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table" 
    IMAGE = "image"

class DocumentChunk(BaseModel):
    id: str
    text: str
    page_number: int
    content_type: ContentType
    metadata: Dict[str, Any] = {}

class ExtractedData(BaseModel):
    filename: str
    chunks: List[DocumentChunk]
    total_pages: int
    metadata: Dict[str, Any] = {}

class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    
class ChatResponse(BaseModel):
    message: str
    sources: List[Dict[str, Any]] = []

class DocumentMetadata(BaseModel):
    filename: str
    status: str
    message: str
    chunks_count: Optional[int] = None
    
class VectorSearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]