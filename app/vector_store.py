import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from loguru import logger

from app.models import VectorSearchResult

class VectorStore:
    def __init__(self, qdrant_url: str, api_key: Optional[str] = None):
        self.qdrant_url = qdrant_url
        self.api_key = api_key
        self.client = None
        self.collection_name = "rag_documents"
        self.embedding_dimension = 384  # for all-MiniLM-L6-v2
        
    async def initialize(self):
        """Initialize Qdrant client and create collection"""
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.api_key
            )
            
            # Check if collection exists, create if not
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            # For development, we can use in-memory Qdrant
            logger.info("Falling back to in-memory Qdrant client")
            self.client = QdrantClient(":memory:")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
    
    async def add_document(
        self, 
        text: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> str:
        """Add a document chunk to the vector store"""
        try:
            point_id = hash(text + metadata.get("filename", "") + str(metadata.get("chunk_id", "")))
            point_id = abs(point_id)  # Ensure positive ID
            
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": text,
                    **metadata
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.debug(f"Added document chunk to vector store: {point_id}")
            return str(point_id)
            
        except Exception as e:
            logger.error(f"Error adding document to vector store: {e}")
            raise
    
    async def search_similar(
        self, 
        query_embedding: List[float], 
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[VectorSearchResult]:
        """Search for similar documents"""
        try:
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold
            )
            
            results = []
            for point in search_result:
                results.append(VectorSearchResult(
                    text=point.payload.get("text", ""),
                    score=point.score,
                    metadata={
                        k: v for k, v in point.payload.items() 
                        if k != "text"
                    }
                ))
            
            logger.debug(f"Found {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector store: {e}")
            raise
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """List all unique documents in the vector store"""
        try:
            # Get all points with scroll
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000
            )
            
            # Group by filename
            documents = {}
            for point in points:
                filename = point.payload.get("filename", "unknown")
                if filename not in documents:
                    documents[filename] = {
                        "filename": filename,
                        "chunks_count": 0,
                        "content_types": set()
                    }
                
                documents[filename]["chunks_count"] += 1
                content_type = point.payload.get("content_type", "text")
                documents[filename]["content_types"].add(content_type)
            
            # Convert sets to lists for JSON serialization
            for doc in documents.values():
                doc["content_types"] = list(doc["content_types"])
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Error listing documents: {e}")
            raise
    
    async def delete_document(self, filename: str) -> int:
        """Delete all chunks of a document"""
        try:
            # Find all points with the given filename
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="filename",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                ),
                limit=1000
            )
            
            if not points:
                return 0
            
            # Delete all found points
            point_ids = [point.id for point in points]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=point_ids)
            )
            
            logger.info(f"Deleted {len(point_ids)} chunks for document: {filename}")
            return len(point_ids)
            
        except Exception as e:
            logger.error(f"Error deleting document {filename}: {e}")
            raise
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance": info.config.params.vectors.distance.value
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}