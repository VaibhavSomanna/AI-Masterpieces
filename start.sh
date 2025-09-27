#!/bin/bash

# RAG Chat Application Startup Script

echo "🚀 Starting RAG Chat Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download additional models if needed
echo "Downloading language models..."
python -c "
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import os

# Download embedding model
print('Downloading embedding model...')
model = SentenceTransformer('all-MiniLM-L6-v2')

# Download generation model
print('Downloading generation model...')
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
model = AutoModel.from_pretrained('microsoft/DialoGPT-small')

print('Models downloaded successfully!')
"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your configuration"
fi

# Start Qdrant in Docker if not running
if ! docker ps | grep -q "qdrant"; then
    echo "Starting Qdrant vector database..."
    docker run -d --name qdrant \
        -p 6333:6333 \
        -v $(pwd)/qdrant_storage:/qdrant/storage \
        qdrant/qdrant
    
    # Wait for Qdrant to start
    echo "Waiting for Qdrant to start..."
    sleep 10
fi

# Start the FastAPI application
echo "🎉 Starting the application on http://localhost:8000"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload