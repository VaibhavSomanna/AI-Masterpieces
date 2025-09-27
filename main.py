from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn
from typing import List, Optional, AsyncGenerator
import json
import asyncio
from pathlib import Path
import os
from loguru import logger

from app.models import ChatRequest, ChatResponse, DocumentMetadata
from app.document_processor import DocumentProcessor
from app.vector_store import VectorStore
from app.chat_service import ChatService
from app.config import Settings

# Initialize settings
settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat Application",
    description="A comprehensive RAG system with support for PDFs, images, and tables",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize services
document_processor = DocumentProcessor()
vector_store = VectorStore(settings.qdrant_url, settings.qdrant_api_key)
chat_service = ChatService(vector_store, settings.model_name)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting RAG Chat Application...")
    await vector_store.initialize()
    await chat_service.initialize()
    logger.info("Services initialized successfully!")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the main chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RAG Chat Application is running"}

@app.post("/upload", response_model=DocumentMetadata)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and process a document (PDF, image, or document with tables)"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # Save uploaded file
    upload_path = Path("uploads") / file.filename
    upload_path.parent.mkdir(exist_ok=True)
    
    try:
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process document in background
        background_tasks.add_task(process_document, str(upload_path), file.filename)
        
        return DocumentMetadata(
            filename=file.filename,
            status="processing",
            message="Document uploaded successfully and is being processed"
        )
    
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

async def process_document(file_path: str, filename: str):
    """Background task to process uploaded document"""
    try:
        logger.info(f"Processing document: {filename}")
        
        # Extract text and metadata
        extracted_data = await document_processor.process_document(file_path)
        
        # Create embeddings and store in vector database
        for chunk in extracted_data.chunks:
            embedding = await chat_service.create_embedding(chunk.text)
            await vector_store.add_document(
                chunk.text,
                embedding,
                {
                    "filename": filename,
                    "chunk_id": chunk.id,
                    "page_number": chunk.page_number,
                    "content_type": chunk.content_type
                }
            )
        
        logger.info(f"Document processed successfully: {filename}")
    
    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")

@app.post("/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """Stream chat responses using RAG"""
    try:
        async def generate_response() -> AsyncGenerator[str, None]:
            async for chunk in chat_service.stream_chat_response(chat_request.message):
                yield f"data: {json.dumps({'content': chunk, 'type': 'content'})}\n\n"
            yield f"data: {json.dumps({'type': 'end'})}\n\n"
        
        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
    
    except Exception as e:
        logger.error(f"Error in chat stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    """Non-streaming chat endpoint"""
    try:
        response = await chat_service.get_chat_response(chat_request.message)
        return ChatResponse(
            message=response,
            sources=chat_service.last_sources
        )
    
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        documents = await vector_store.list_documents()
        return {"documents": documents}
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document and its embeddings"""
    try:
        deleted_count = await vector_store.delete_document(filename)
        return {"message": f"Deleted {deleted_count} chunks for document: {filename}"}
    
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )