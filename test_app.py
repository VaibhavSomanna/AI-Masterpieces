#!/usr/bin/env python3
"""
Simple test script to verify FastAPI application works without all dependencies
"""

import sys
import os
sys.path.append('.')

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# Simple test app
test_app = FastAPI(title="RAG Test Application")

# Mount templates
try:
    templates = Jinja2Templates(directory="templates")
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    test_app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {e}")
    templates = None

@test_app.get("/")
async def root():
    return {"message": "RAG Chat Application is running!", "status": "healthy"}

@test_app.get("/health")
async def health():
    return {"status": "healthy", "components": {"api": "ok", "templates": templates is not None}}

@test_app.get("/chat", response_class=HTMLResponse)
async def chat_interface(request: Request):
    if templates:
        try:
            return templates.TemplateResponse("chat.html", {"request": request})
        except Exception as e:
            return HTMLResponse(f"<h1>RAG Chat Application</h1><p>Template error: {e}</p><p>API is working at /health</p>")
    else:
        return HTMLResponse("<h1>RAG Chat Application</h1><p>Templates not available, but API is working at /health</p>")

@test_app.post("/upload")
async def test_upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "status": "would be processed",
        "message": f"File {file.filename} received successfully"
    }

if __name__ == "__main__":
    print("🚀 Starting RAG Chat Application Test Server...")
    print("📋 Testing basic FastAPI functionality...")
    print("🌐 Will be available at: http://localhost:8000")
    print("📊 Health check: http://localhost:8000/health")
    print("💬 Chat interface: http://localhost:8000/chat")
    
    uvicorn.run(test_app, host="0.0.0.0", port=8000, log_level="info")