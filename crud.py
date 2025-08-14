#crud.py → calls vector_db for embedding storage, keeping PostgreSQL + Pinecone logic in sync.
from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional, Dict, Any
from datetime import datetime, date, timedelta
import logging

from . import models, schemas
from .vector_db import vector_db

logger = logging.getLogger(__name__)


# ============================================================================
# FOOD LOG CRUD WITH PINECONE INTEGRATION
# ============================================================================

async def create_food_log(
    db: Session, 
    user_id: int, 
    food_log: schemas.CreateFoodLog
) -> models.FoodLog:
    """Create a new food log with PostgreSQL + Pinecone integration."""
    
    # 1. Insert structured data into PostgreSQL
    db_food_log = models.FoodLog(
        user_id=user_id,
        food_name=food_log.food_name,
        quantity_g=food_log.quantity_g,
        calories_total=food_log.calories_total,
        protein_g=food_log.protein_g,
        carbs_g=food_log.carbs_g,
        fat_g=food_log.fat_g,
        meal_type=food_log.meal_type,
        meal_date=food_log.meal_date,
        photo_url=food_log.photo_url
    )
    
    db.add(db_food_log)
    db.commit()
    db.refresh(db_food_log)
    
    # 2. Generate embedding and store in Pinecone
    try:
        vector_id = await vector_db.upsert_food_log(
            food_log_id=db_food_log.id,
            user_id=user_id,
            food_name=food_log.food_name,
            meal_type=food_log.meal_type,
            meal_date=food_log.meal_date,
            quantity_g=food_log.quantity_g,
            calories_total=food_log.calories_total
        )
        
        # 3. Update PostgreSQL record with embedding info
        db_food_log.has_embedding = True
        db_food_log.embedding_id = vector_id
        db.commit()
        db.refresh(db_food_log)
        
        logger.info(f"Created food log {db_food_log.id} with embedding {vector_id}")
        
    except Exception as e:
        logger.error(f"Failed to create embedding for food log {db_food_log.id}: {str(e)}")
        # Don't fail the entire operation if embedding fails
    
    return db_food_log

async def search_similar_food_logs(
    db: Session,
    query: str,
    user_id: Optional[int] = None,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Search for similar food logs using Pinecone similarity search."""
    
    # 1. Search Pinecone for similar vectors
    similar_vectors = await vector_db.search_similar_foods(
        query_text=query,
        user_id=user_id,
        top_k=limit
    )
    
    if not similar_vectors:
        return []
    
    # 2. Extract food log IDs from Pinecone results
    food_log_ids = [item["food_log_id"] for item in similar_vectors]
    
    # 3. Fetch structured data from PostgreSQL
    db_food_logs = db.query(models.FoodLog).filter(
        models.FoodLog.id.in_(food_log_ids)
    ).all()
    
    # 4. Combine Pinecone similarity scores with PostgreSQL data
    results = []
    food_log_dict = {fl.id: fl for fl in db_food_logs}
    
    for vector_item in similar_vectors:
        food_log_id = vector_item["food_log_id"]
        if  food_log_id in food_log_dict:
            food_log = food_log_dict[food_log_id]
            results.append({
                "food_log": food_log,
                "similarity_score": vector_item["score"],
                "pinecone_metadata": vector_item["metadata"]
            })
    
    return results

def get_exercise_logs(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 10,
    date_filter: Optional[date] = None
) -> tuple[List[models.ExerciseLog], int]:
    """Get exercise logs with optional date filtering."""
    
    query = db.query(models.ExerciseLog).filter(models.ExerciseLog.user_id == user_id)
    
    if date_filter:
        query = query.filter(
            and_(
                models.ExerciseLog.exercise_date >= date_filter,
                models.ExerciseLog.exercise_date < date_filter + timedelta(days=1)
            )
        )
    
    total_count = query.count()
    exercise_logs = query.order_by(models.ExerciseLog.exercise_date.desc()).offset(skip).limit(limit).all()
    
    return exercise_logs, total_count

async def delete_exercise_log(db: Session, exercise_log_id: int, user_id: int) -> bool:
    """Delete exercise log from both PostgreSQL and Pinecone."""
    
    # 1. Find the exercise log
    exercise_log = db.query(models.ExerciseLog).filter(
        and_(
            models.ExerciseLog.id == exercise_log_id,
            models.ExerciseLog.user_id == user_id
        )
    ).first()
    
    if not exercise_log:
        return False
    
    # 2. Delete from Pinecone if embedding exists
    if exercise_log.has_embedding and exercise_log.embedding_id:
        try:
            await vector_db.delete_vector(exercise_log.embedding_id)
        except Exception as e:
            logger.warning(f"Failed to delete vector {exercise_log.embedding_id}: {str(e)}")
    
    # 3. Delete from PostgreSQL
    db.delete(exercise_log)
    db.commit()
    
    logger.info(f"Deleted exercise log {exercise_log_id}")
    return {
    "success": True,
    "score": vector_item["score"],
    "pinecone_metadata": vector_item["metadata"]
    }

def get_food_logs(
    db: Session,
    user_id: int,
    skip: int = 0,
    limit: int = 10,
    date_filter: Optional[date] = None
) -> tuple[List[models.FoodLog], int]:
    """Get food logs with optional date filtering."""
    
    query = db.query(models.FoodLog).filter(models.FoodLog.user_id == user_id)
    
    if date_filter:
        query = query.filter(
            and_(
                models.FoodLog.meal_date >= date_filter,
                models.FoodLog.meal_date < date_filter + timedelta(days=1)
            )
        )
    
    total_count = query.count()
    food_logs = query.order_by(models.FoodLog.meal_date.desc()).offset(skip).limit(limit).all()
    
    return food_logs, total_count

async def delete_food_log(db: Session, food_log_id: int, user_id: int) -> bool:
    """Delete food log from both PostgreSQL and Pinecone."""
    
    # 1. Find the food log
    food_log = db.query(models.FoodLog).filter(
        and_(
            models.FoodLog.id == food_log_id,
            models.FoodLog.user_id == user_id
        )
    ).first()
    
    if not food_log:
        return False
    
    # 2. Delete from Pinecone if embedding exists
    if food_log.has_embedding and food_log.embedding_id:
        try:
            await vector_db.delete_vector(food_log.embedding_id)
        except Exception as e:
            logger.warning(f"Failed to delete vector {food_log.embedding_id}: {str(e)}")
    
    # 3. Delete from PostgreSQL
    db.delete(food_log)
    db.commit()
    
    logger.info(f"Deleted food log {food_log_id}")
    return True

# ============================================================================
# EXERCISE LOG CRUD WITH PINECONE INTEGRATION
# ============================================================================

async def create_exercise_log(
    db: Session,
    user_id: int,
    exercise_log: schemas.CreateExerciseLog
) -> models.ExerciseLog:
    """Create a new exercise log with PostgreSQL + Pinecone integration."""
    
    # 1. Insert structured data into PostgreSQL
    db_exercise_log = models.ExerciseLog(
        user_id=user_id,
        exercise_name=exercise_log.exercise_name,
        exercise_type=exercise_log.exercise_type,
        duration_minutes=exercise_log.duration_minutes,
        calories_burned=exercise_log.calories_burned,
        intensity_level=exercise_log.intensity_level,
        exercise_date=exercise_log.exercise_date,
        notes=exercise_log.notes
    )
    
    db.add(db_exercise_log)
    db.commit()
    db.refresh(db_exercise_log)
    
    # 2. Generate embedding and store in Pinecone
    try:
        vector_id = await vector_db.upsert_exercise_log(
            exercise_log_id=db_exercise_log.id,
            user_id=user_id,
            exercise_name=exercise_log.exercise_name,
            exercise_type=exercise_log.exercise_type,
            exercise_date=exercise_log.exercise_date,
            duration_minutes=exercise_log.duration_minutes,
            calories_burned=exercise_log.calories_burned
        )
        
        # 3. Update PostgreSQL record with embedding info
        db_exercise_log.has_embedding = True
        db_exercise_log.embedding_id = vector_id
        db.commit()
        db.refresh(db_exercise_log)
        
        logger.info(f"Created exercise log {db_exercise_log.id} with embedding {vector_id}")
        
    except Exception as e:
        logger.error(f"Failed to create embedding for exercise log {db_exercise_log.id}: {str(e)}")
    
    return db_exercise_log

async def search_similar_exercise_logs(
    db: Session,
    query: str,
    user_id: Optional[int] = None,
    limit: int = 10,
    min_score: float = 0.0
) -> List[Dict[str, Any]]:
    """Search for similar exercise logs using Pinecone similarity search."""
    
    # 1. Search Pinecone for similar vectors
    similar_vectors = await vector_db.search_similar_exercises(
        query_text=query,
        user_id=user_id,
        top_k=limit,
        similarity_threshold=min_score
    )
    
    if not similar_vectors:
        return []
    
    # 2. Extract exercise log IDs from Pinecone results
    exercise_log_ids = [item["exercise_log_id"] for item in similar_vectors]
    
    # 3. Fetch structured data from PostgreSQL
    db_exercise_logs = db.query(models.ExerciseLog).filter(
        models.ExerciseLog.id.in_(exercise_log_ids)
    ).all()
    
    # 4. Combine results
    results = []
    exercise_log_dict = {el.id: el for el in db_exercise_logs}
    
    for vector_item in similar_vectors:
        exercise_log_id = vector_item["exercise_log_id"]
        if exercise_log_id in exercise_log_dict:
            exercise_log = exercise_log_dict[exercise_log_id]
            results.append({
                "exercise_log": exercise_log,
                "similarity_score": vector_item["score"],
                "pinecone_metadata": vector_item["metadata"]        # metadata stored in Pinecone
        })
    return results