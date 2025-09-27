# AI-Masterpieces

This repository will contain numerous projects that can be used in real time applications showcasing the diverse features of what AI is capable in todays world.

## RAG Chat Application

A comprehensive Retrieval Augmented Generation (RAG) system built with FastAPI that supports multiple document types and provides intelligent, context-aware responses with source citations.

### ✨ Features

- **Multi-format Document Support**: Process PDFs, Word documents, text files, and images
- **Advanced Text Extraction**: 
  - PDF text and table extraction using PyPDF2 and Tabula
  - Image OCR using Tesseract
  - Word document processing with python-docx
- **Vector Embeddings**: Sentence Transformers for high-quality semantic embeddings
- **Vector Database**: Qdrant integration for efficient similarity search
- **Streaming Chat**: Real-time streaming responses for better user experience
- **Source Citations**: Automatic source attribution with confidence scores
- **Modern UI**: Beautiful, responsive web interface with drag-and-drop upload
- **RESTful API**: Clean API endpoints for integration with other applications

### 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Interface │────│   FastAPI App    │────│  Qdrant Vector  │
│   (HTML/JS/CSS) │    │                  │    │    Database     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                       ┌───────┴───────┐
                       │               │
                ┌──────▼──────┐ ┌──────▼──────┐
                │  Document   │ │    Chat     │
                │  Processor  │ │   Service   │
                └─────────────┘ └─────────────┘
                       │               │
                ┌──────▼──────┐ ┌──────▼──────┐
                │ Sentence    │ │ HuggingFace │
                │Transformers │ │   Models    │
                └─────────────┘ └─────────────┘
```

### 🚀 Quick Start

#### Prerequisites
- Python 3.8+
- Docker (for Qdrant vector database)
- Tesseract OCR (for image text extraction)

#### Installation & Setup

1. **Clone the repository**:
```bash
git clone https://github.com/VaibhavSomanna/AI-Masterpieces.git
cd AI-Masterpieces
```

2. **Run the startup script**:
```bash
./start.sh
```

This script will:
- Create a virtual environment
- Install all dependencies
- Download required ML models
- Start Qdrant vector database in Docker
- Launch the FastAPI application

3. **Access the application**:
- Open your browser and go to: http://localhost:8000
- You'll see the chat interface ready to use!

#### Manual Setup (Alternative)

1. **Install dependencies**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Install Tesseract OCR**:
   - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

3. **Start Qdrant**:
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env file with your settings
```

5. **Run the application**:
```bash
uvicorn main:app --reload
```

### 📚 Usage

#### Uploading Documents

1. **Drag and drop** files onto the upload area, or **click to browse**
2. Supported formats:
   - **PDFs**: Text extraction + table detection
   - **Word docs** (.docx): Text + table extraction
   - **Text files** (.txt): Direct text processing
   - **Images** (.png, .jpg, .jpeg): OCR text extraction

3. Documents are processed automatically and added to the vector database

#### Chatting with Documents

1. **Ask questions** about your uploaded documents
2. The system will:
   - Find relevant content using semantic search
   - Generate contextual responses
   - Provide source citations with confidence scores
3. **Streaming responses** provide real-time feedback

#### Example Queries

- *"What are the main findings in the research paper?"*
- *"Summarize the financial data from the quarterly report"*
- *"What does the contract say about payment terms?"*
- *"Extract the key dates mentioned in the document"*

### 🛠️ API Endpoints

#### Document Management
- `POST /upload` - Upload and process documents
- `GET /documents` - List all processed documents
- `DELETE /documents/{filename}` - Remove a document

#### Chat Interface
- `POST /chat` - Get a complete chat response
- `POST /chat/stream` - Get streaming chat response
- `GET /` - Access the web interface

#### System
- `GET /health` - Health check endpoint

### ⚙️ Configuration

Edit the `.env` file to customize:

```bash
# Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=rag_documents

# Models
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
HF_MODEL_NAME=microsoft/DialoGPT-small

# Processing
MAX_CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE=52428800  # 50MB
```

### 🔧 Development

#### Project Structure
```
├── main.py              # FastAPI application entry point
├── app/
│   ├── models.py        # Pydantic models
│   ├── config.py        # Configuration settings
│   ├── document_processor.py  # Document processing logic
│   ├── vector_store.py  # Qdrant integration
│   └── chat_service.py  # RAG and chat logic
├── templates/
│   └── chat.html        # Web interface
├── static/              # Static assets (CSS, JS, images)
├── uploads/             # Uploaded files storage
└── requirements.txt     # Python dependencies
```

#### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/
```

### 🎯 Advanced Features

#### Custom Models
Replace the default models by updating the configuration:
- **Embedding models**: Any SentenceTransformer model
- **Generation models**: Any HuggingFace text generation model
- **OpenAI integration**: Set `OPENAI_API_KEY` for GPT models

#### Production Deployment
- Use `gunicorn` for production ASGI server
- Configure reverse proxy (nginx)
- Set up proper logging and monitoring
- Use persistent storage for Qdrant

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

### 🙏 Acknowledgments

- **Qdrant** for the excellent vector database
- **Sentence Transformers** for semantic embeddings
- **HuggingFace** for transformer models
- **FastAPI** for the amazing web framework
