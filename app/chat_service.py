import asyncio
from typing import List, Dict, Any, AsyncGenerator
import json
from transformers import AutoTokenizer, AutoModel, pipeline
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from loguru import logger

from app.vector_store import VectorStore
from app.models import VectorSearchResult

class ChatService:
    def __init__(self, vector_store: VectorStore, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.vector_store = vector_store
        self.embedding_model_name = embedding_model_name
        self.embedding_model = None
        self.generation_model = None
        self.tokenizer = None
        self.last_sources = []
        
    async def initialize(self):
        """Initialize the embedding and generation models"""
        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Load generation model (using a lightweight model for demo)
            logger.info("Loading generation model...")
            model_name = "microsoft/DialoGPT-small"  # Lightweight model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add pad token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Use text generation pipeline for easier streaming
            self.generation_model = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                max_length=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    async def create_embedding(self, text: str) -> List[float]:
        """Create embedding for text"""
        try:
            # Run embedding in executor to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, 
                self._create_embedding_sync, 
                text
            )
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            raise
    
    def _create_embedding_sync(self, text: str) -> np.ndarray:
        """Synchronous embedding creation"""
        return self.embedding_model.encode(text)
    
    async def get_chat_response(self, user_message: str) -> str:
        """Get a complete chat response using RAG"""
        try:
            # Create embedding for user message
            query_embedding = await self.create_embedding(user_message)
            
            # Search for relevant documents
            search_results = await self.vector_store.search_similar(
                query_embedding, 
                limit=3,
                score_threshold=0.3
            )
            
            # Store sources for reference
            self.last_sources = [
                {
                    "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                    "score": result.score,
                    "filename": result.metadata.get("filename", "unknown"),
                    "page_number": result.metadata.get("page_number", 1),
                    "content_type": result.metadata.get("content_type", "text")
                }
                for result in search_results
            ]
            
            # Create context from search results
            context = self._create_context(search_results)
            
            # Generate response
            prompt = self._create_prompt(user_message, context)
            response = await self._generate_response(prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting chat response: {e}")
            return "I apologize, but I encountered an error while processing your request. Please try again."
    
    async def stream_chat_response(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream chat response using RAG"""
        try:
            # Create embedding for user message
            query_embedding = await self.create_embedding(user_message)
            
            # Search for relevant documents
            search_results = await self.vector_store.search_similar(
                query_embedding, 
                limit=3,
                score_threshold=0.3
            )
            
            # Store sources for reference
            self.last_sources = [
                {
                    "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                    "score": result.score,
                    "filename": result.metadata.get("filename", "unknown"),
                    "page_number": result.metadata.get("page_number", 1),
                    "content_type": result.metadata.get("content_type", "text")
                }
                for result in search_results
            ]
            
            # Create context from search results
            context = self._create_context(search_results)
            
            # Generate streaming response
            prompt = self._create_prompt(user_message, context)
            
            async for chunk in self._stream_response(prompt):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error streaming chat response: {e}")
            yield "I apologize, but I encountered an error while processing your request."
    
    def _create_context(self, search_results: List[VectorSearchResult]) -> str:
        """Create context from search results"""
        if not search_results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            context_parts.append(
                f"Document {i} (from {result.metadata.get('filename', 'unknown')}, "
                f"page {result.metadata.get('page_number', 1)}):\n{result.text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, user_message: str, context: str) -> str:
        """Create prompt for the generation model"""
        return f"""Based on the following context, please answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

User Question: {user_message}

Answer:"""
    
    async def _generate_response(self, prompt: str) -> str:
        """Generate a complete response"""
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._generate_response_sync,
                prompt
            )
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
    
    def _generate_response_sync(self, prompt: str) -> str:
        """Synchronous response generation"""
        try:
            # Truncate prompt if too long
            max_prompt_length = 300
            if len(prompt) > max_prompt_length:
                prompt = prompt[:max_prompt_length] + "..."
            
            # Generate response
            outputs = self.generation_model(
                prompt,
                max_new_tokens=150,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            # Remove the original prompt from the response
            if prompt in generated_text:
                response = generated_text.replace(prompt, "").strip()
            else:
                response = generated_text.strip()
            
            # If response is empty or too short, provide a fallback
            if len(response) < 10:
                response = "Based on the available information, I can provide some insights, but I'd recommend asking a more specific question for a better answer."
            
            return response
            
        except Exception as e:
            logger.error(f"Error in sync generation: {e}")
            return "I encountered an issue generating a response. Please try again."
    
    async def _stream_response(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream response generation"""
        try:
            # For demonstration, we'll simulate streaming by yielding the response in chunks
            full_response = await self._generate_response(prompt)
            
            # Split response into words for streaming effect
            words = full_response.split()
            for i, word in enumerate(words):
                if i == 0:
                    yield word
                else:
                    yield f" {word}"
                
                # Small delay to simulate streaming
                await asyncio.sleep(0.05)
                
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            yield "Error generating streaming response."