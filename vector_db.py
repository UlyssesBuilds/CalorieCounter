# vector_db.py - Pinecone vector database operations with Hugging Face API
import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import httpx
from pinecone import Pinecone, ServerlessSpec
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self):
        """Initialize Pinecone client and Hugging Face client."""
        self.pc = None
        self.index = None
        self.hf_client = None
        self._initialized = False
        
        # Configuration
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "fitness-app-vectors")
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        # Embedding model configuration
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL")
        self.embedding_dimension = 1024  # all-MiniLM-L6-v2 dimension
        
    async def initialize(self):
        """Initialize Pinecone connection and Hugging Face client."""
        if self._initialized:
            return
            
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Create index if it doesn't exist
            await self._ensure_index_exists()
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Initialize Hugging Face client
            self.hf_client = InferenceClient(
                model=self.embedding_model_name,
                token=self.hf_token
            )
            
            self._initialized = True
            logger.info(f"Vector database initialized with index: {self.index_name}")
            logger.info(f"Using Hugging Face model: {self.embedding_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            raise
    
    async def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist."""
        loop = asyncio.get_event_loop()
        
        def _check_and_create_index():
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.environment
                    )
                )
                # Wait for index to be ready
                import time
                time.sleep(10)
            
        await loop.run_in_executor(None, _check_and_create_index)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Hugging Face API."""
        if not self._initialized:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            
            def _get_embedding():
                """Synchronous function to call HF API."""
                # Use the feature extraction task for embeddings
                embedding = self.hf_client.feature_extraction(text)
                
                # Handle different response formats from HF API
                if isinstance(embedding, list):
                    # If it's a list of lists, take the first one
                    if isinstance(embedding[0], list):
                        embedding = embedding[0]
                elif hasattr(embedding, 'tolist'):
                    # If it's a numpy array or tensor
                    embedding = embedding.tolist()
                
                return embedding
            
            embedding = await loop.run_in_executor(None, _get_embedding)
            
            # Validate embedding dimension
            if len(embedding) != self.embedding_dimension:
                logger.warning(f"Unexpected embedding dimension: {len(embedding)}, expected: {self.embedding_dimension}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for text '{text[:50]}...': {str(e)}")
            # Fallback: return zero vector
            return [0.0] * self.embedding_dimension
    
    async def upsert_food_log(
        self,
        food_log_id: int,
        user_id: int,
        food_name: str,
        meal_type: Optional[str] = None,
        meal_date: Optional[datetime] = None,
        quantity_g: Optional[float] = None,
        calories_total: Optional[float] = None
    ) -> str:
        """Upsert a food log vector to Pinecone."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generate embedding using HF API
            embedding = await self.generate_embedding(food_name)
            
            # Create vector ID
            vector_id = f"foodlog:{food_log_id}"
            
            # Prepare metadata
            metadata = {
                "user_id": user_id,
                "food_name": food_name,
                "type": "food_log",
                "meal_type": meal_type or "unknown",
                "meal_date": meal_date.isoformat() if meal_date else None,
                "quantity_g": quantity_g,
                "calories_total": calories_total,
                "created_at": datetime.now().isoformat()
            }
            
            # Clean metadata (remove None values)
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Upsert to Pinecone
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.index.upsert(vectors=[(vector_id, embedding, metadata)])
            )
            
            logger.info(f"Upserted food log vector: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"Failed to upsert food log {food_log_id}: {str(e)}")
            raise
    
    async def upsert_exercise_log(
        self,
        exercise_log_id: int,
        user_id: int,
        exercise_name: str,
        exercise_type: Optional[str] = None,
        exercise_date: Optional[datetime] = None,
        duration_minutes: Optional[int] = None,
        calories_burned: Optional[float] = None
    ) -> str:
        """Upsert an exercise log vector to Pinecone."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generate embedding using HF API
            embedding_text = f"{exercise_name} {exercise_type or ''}"
            embedding = await self.generate_embedding(embedding_text.strip())
            
            # Create vector ID
            vector_id = f"exercise:{exercise_log_id}"
            
            # Prepare metadata
            metadata = {
                "user_id": user_id,
                "exercise_name": exercise_name,
                "exercise_type": exercise_type or "unknown",
                "type": "exercise_log",
                "exercise_date": exercise_date.isoformat() if exercise_date else None,
                "duration_minutes": duration_minutes,
                "calories_burned": calories_burned,
                "created_at": datetime.now().isoformat()
            }
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Upsert to Pinecone
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.index.upsert(vectors=[(vector_id, embedding, metadata)])
            )
            
            logger.info(f"Upserted exercise log vector: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"Failed to upsert exercise log {exercise_log_id}: {str(e)}")
            raise
    
    async def search_similar_foods(
        self,
        query_text: str,
        user_id: Optional[int] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar food logs."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generate query embedding using HF API
            query_embedding = await self.generate_embedding(query_text)
            
            # Prepare filter
            filter_dict = {"type": {"$eq": "food_log"}}
            if user_id:
                filter_dict["user_id"] = {"$eq": user_id}
                
            # Search in Pinecone
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_embedding,
                    filter=filter_dict,
                    top_k=top_k,
                    include_metadata=True
                )
            )
            
            # Process results
            similar_foods = []
            for match in results.matches:
                if match.score >= similarity_threshold:
                    similar_foods.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata,
                        "food_log_id": int(match.id.split(":")[1])  # Extract ID from "foodlog:123"
                    })
                    
            logger.info(f"Found {len(similar_foods)} similar foods for query: {query_text}")
            return similar_foods
            
        except Exception as e:
            logger.error(f"Failed to search similar foods: {str(e)}")
            return []
    
    async def search_similar_exercises(
        self,
        query_text: str,
        user_id: Optional[int] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar exercise logs."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generate query embedding using HF API
            query_embedding = await self.generate_embedding(query_text)
            
            # Prepare filter
            filter_dict = {"type": {"$eq": "exercise_log"}}
            if user_id:
                filter_dict["user_id"] = {"$eq": user_id}
                
            # Search in Pinecone
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_embedding,
                    filter=filter_dict,
                    top_k=top_k,
                    include_metadata=True
                )
            )
            
            # Process results
            similar_exercises = []
            for match in results.matches:
                if match.score >= similarity_threshold:
                    similar_exercises.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata,
                        "exercise_log_id": int(match.id.split(":")[1])  # Extract ID from "exercise:123"
                    })
                    
            logger.info(f"Found {len(similar_exercises)} similar exercises for query: {query_text}")
            return similar_exercises
            
        except Exception as e:
            logger.error(f"Failed to search similar exercises: {str(e)}")
            return []
    
    async def delete_vector(self, vector_id: str) -> bool:
        """Delete a vector from Pinecone."""
        if not self._initialized:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.index.delete(ids=[vector_id])
            )
            logger.info(f"Deleted vector: {vector_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {str(e)}")
            return False
    
    async def get_index_stats(self) -> Dict[str, Any]:
        """Get Pinecone index statistics."""
        if not self._initialized:
            await self.initialize()
            
        try:
            loop = asyncio.get_event_loop()
            stats = await loop.run_in_executor(
                None,
                lambda: self.index.describe_index_stats()
            )
            return stats
        except Exception as e:
            logger.error(f"Failed to get index stats: {str(e)}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the vector database."""
        try:
            if not self._initialized:
                await self.initialize()
                
            # Test embedding generation
            test_embedding = await self.generate_embedding("health check")
            
            # Get index stats
            stats = await self.get_index_stats()
            
            return {
                "status": "healthy",
                "initialized": self._initialized,
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": len(test_embedding),
                "index_name": self.index_name,
                "pinecone_stats": stats
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }

# Global instance
vector_db = VectorDatabase()