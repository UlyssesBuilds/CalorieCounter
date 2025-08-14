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
        """Initialize Pinecone client."""
        self.pc = None
        self.index = None
        self._initialized = False
        
        # Configuration
        self.api_key = os.getenv("PINECONE_API_KEY")
        #self.environment = os.getenv("PINECONE_ENVIRONMENT")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        
        # Embedding model configuration
        self.embedding_dimension = 1024  # Pinecone model for embedding
        
    async def initialize(self):
        """Initialize Pinecone connection."""
        if self._initialized:
            return
            
        try:
            # Initialize Pinecone client
            self.pc = Pinecone(api_key=self.api_key)
            
            # Connect to your existing index
            self.index = self.pc.Index(self.index_name)
            
            self._initialized = True
            logger.info(f"Vector database initialized with existing index: {self.index_name}")
            logger.info(f"Using Pinecone's llama-text-embed-v2 model (1024 dimensions)")
            
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
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using Pinecone's inference API."""
        try:
            loop = asyncio.get_event_loop()
            embedding_response = await loop.run_in_executor(
                None,
                lambda: self.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[{"text": text}],
                    parameters={"input_type": "query", "truncate": "END"}
                )
            )
            return embedding_response[0]['values']
        except Exception as e:
            logger.error(f"Failed to generate embedding for text '{text}': {str(e)}")
            raise
    
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
        """Upsert a food log vector to Pinecone using built-in embeddings."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Create vector ID
            vector_id = f"foodlog:{food_log_id}"
            
            # 1. BUILD COMPREHENSIVE TEXT for Pinecone's embedding model
            text_parts = [food_name]
            
            if meal_type:
                text_parts.append(f"meal type: {meal_type}")
            if quantity_g:
                text_parts.append(f"quantity: {quantity_g}g")
            if calories_total:
                text_parts.append(f"calories: {calories_total}")
            embedding_text = " ".join(text_parts)
            
            # Prepare metadata
            metadata = {
                "user_id": int(user_id),
                "food_name": str(food_name),
                "type": "food_log",
                "meal_type": str(meal_type) if meal_type else "unknown",
                "meal_date": meal_date.isoformat() if meal_date else None,
                "quantity_g": float(quantity_g) if quantity_g is not None else None,
                "calories_total": float(calories_total) if calories_total is not None else None,
                "created_at": datetime.now().isoformat()
            }
            
            # Clean metadata (remove None values)
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Step 1: Generate embedding using Pinecone's inference API
            loop = asyncio.get_event_loop()
            embedding_response = await loop.run_in_executor(
                None,
                lambda: self.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[{"text": embedding_text}],
                    parameters={"input_type": "passage", "truncate": "END"}
                )
            )
            
            # Step 2: Extract the embedding vector
            embedding_vector = embedding_response[0]['values']
            
            # Step 3: Upsert to Pinecone with the embedding vector
            await loop.run_in_executor(
                None,
                lambda: self.index.upsert(
                    vectors=[{  # ✅ Use 'vectors=' not 'data='
                        "id": vector_id,
                        "values": embedding_vector,  # ✅ Use embedding_vector not embedding_text
                        "metadata": metadata
                    }]
                )
            )
            
            logger.info(f"Successfully upserted food log vector: {vector_id}")
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
        """Upsert an exercise log vector to Pinecone using built-in embeddings."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Create vector ID
            vector_id = f"exercise:{exercise_log_id}"
            
            # Create embedding text
            text_parts = [exercise_name]
            if exercise_type:
                text_parts.append(f"type: {exercise_type}")
            if duration_minutes:
                text_parts.append(f"duration: {duration_minutes} minutes")
            if calories_burned:
                text_parts.append(f"calories burned: {calories_burned}")
            
            embedding_text = " ".join(text_parts)
            
            # Prepare metadata
            metadata = {
                "user_id": int(user_id),
                "exercise_name": str(exercise_name),
                "exercise_type": str(exercise_type) if exercise_type else "unknown",
                "type": "exercise_log",
                "exercise_date": exercise_date.isoformat() if exercise_date else None,
                "duration_minutes": int(duration_minutes) if duration_minutes is not None else None,
                "calories_burned": float(calories_burned) if calories_burned is not None else None,
                "created_at": datetime.now().isoformat()
                
            }
            
            # Clean metadata
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Step 1: Generate embedding using Pinecone's inference API
            loop = asyncio.get_event_loop()
            embedding_response = await loop.run_in_executor(
                None,
                lambda: self.pc.inference.embed(
                    model="llama-text-embed-v2",
                    inputs=[{"text": embedding_text}],
                    parameters={"input_type": "passage", "truncate": "END"}
                )
            )

            # Step 2: Extract the embedding vector
            embedding_vector = embedding_response[0]['values']

            # Step 3: Upsert to Pinecone with the embedding vector
            await loop.run_in_executor(
                None,
                lambda: self.index.upsert(
                    vectors=[{
                        "id": vector_id,
                        "values": embedding_vector,
                        "metadata": metadata
                    }]
                )
            )
            
            logger.info(f"Successfully upserted exercise log vector: {vector_id}")
            return vector_id
            
        except Exception as e:
            logger.error(f"Failed to upsert exercise log {exercise_log_id}: {str(e)}")
            raise
    
    ###############
    async def search_similar_foods(
        self,
        query_text: str,
        user_id: Optional[int] = None,
        top_k: int = 10,
        similarity_threshold: float = 0.3
    ) -> List[Dict[str, Any]]:
        """Search for similar food logs using query enhancement to match storage format."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # 🔍 DEBUG: Let's see what we're actually searching for
            print(f"🔍 Original query: '{query_text}'") 
            # Enhance queries to match your storage format
            # Try multiple approaches simultaneously
            search_variations = [
                query_text,  # Original: "beef"
                f"{query_text} meal type: lunch",  # Your current storage format
                f"{query_text} lunch",  # Simpler format
                f"{query_text} breakfast",
                f"{query_text} dinner"
            ]
            
            all_results = []
            
            for variation in search_variations:
                print(f"🔍 Trying variation: '{variation}'")
                
                loop = asyncio.get_event_loop()
                embedding_response = await loop.run_in_executor(
                    None,
                    lambda v=variation: self.pc.inference.embed(
                        model="llama-text-embed-v2",
                        inputs=[{"text": v}],
                        parameters={"input_type": "query", "truncate": "END"}
                    )
                )
                
                query_vector = embedding_response[0]['values']
                
                filter_dict = {"type": {"$eq": "food_log"}}
                if user_id:
                    filter_dict["user_id"] = {"$eq": user_id}
                
                results = await loop.run_in_executor(
                    None,
                    lambda: self.index.query(
                        vector=query_vector,
                        filter=filter_dict,
                        top_k=5,  # Get fewer per variation
                        include_metadata=True
                    )
                )
                
                print(f"📊 Found {len(results.matches)} matches for '{variation}'")
                for match in results.matches:
                    print(f"   Score: {match.score:.3f}, Food: {match.metadata.get('food_name')}, ID: {match.id}")
                
                all_results.extend(results.matches)
            
            # Rest of your existing deduplication logic...
            seen = {}
            for match in all_results:
                if match.score >= similarity_threshold:
                    if match.id not in seen or match.score > seen[match.id].score:
                        seen[match.id] = match
            
            similar_foods = []
            for match in sorted(seen.values(), key=lambda x: x.score, reverse=True)[:top_k]:
                similar_foods.append({
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata,
                    "food_log_id": int(match.id.split(":")[1])
                })
                
            print(f"🎯 Final results: {len(similar_foods)} foods")
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
        """Search for similar exercise logs using Pinecone's built-in embeddings."""
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generate embedding for query text
            query_vector = await self._generate_embedding(query_text)
            
            # Prepare filter
            filter_dict = {"type": {"$eq": "exercise_log"}}
            if user_id:
                filter_dict["user_id"] = {"$eq": user_id}
                
            # Search in Pinecone using the embedding vector
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.index.query(
                    vector=query_vector,  # ✅ Use embedding vector not text
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
                
            # Get index stats to verify connection
            stats = await self.get_index_stats()
            
            return {
                "status": "healthy",
                "initialized": self._initialized,
                "embedding_model": "llama-text-embed-v2 (Pinecone hosted)",
                "embedding_dimension": self.embedding_dimension,
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