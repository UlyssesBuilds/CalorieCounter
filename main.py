# main.py - Corrected FastAPI app with Pinecone integration
from typing import Union, Optional, List
from fastapi import FastAPI, Depends, HTTPException, status, Query
from . import schemas, models, crud
from .database import engine, SessionLocal
from .vector_db import vector_db
from sqlalchemy.orm import Session
from datetime import datetime, date, timedelta
import logging
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Async context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        await vector_db.initialize()
        logger.info("Vector database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize vector database: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Fitness App API", 
    description="AI-powered fitness tracking API",
    lifespan=lifespan
)

# Database setup - Only for development
models.Base.metadata.drop_all(engine)   # Remove in production
models.Base.metadata.create_all(engine)

def get_db():
    """Database dependency for FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============================================================================
# BASIC ENDPOINTS
# ============================================================================

@app.get("/")
async def read_root():
    return {
        "message": "Welcome to the AI-Powered Fitness API! Visit /docs for documentation.",
        "version": "1.0.0",
        "features": ["food_logging", "exercise_tracking", "ai_search"]
    }

@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": "connected"
    }

# ============================================================================
# USER MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/register", response_model=schemas.UserResponse)
async def register(request: schemas.CreateUser, db: Session = Depends(get_db)):
    """Register a new user."""
    # Check for existing user
    existing_user = db.query(models.User).filter(
        (models.User.email == request.email) | (models.User.username == request.username)
    ).first()
    
    if existing_user:
        detail = "Email already registered" if existing_user.email == request.email else "Username already taken"
        raise HTTPException(status_code=400, detail=detail)
    
    # Create new user
    new_user = models.User(
        email=request.email,
        username=request.username,
        password=request.password,  # TODO: Hash password with bcrypt
        first_name=request.first_name,
        age=request.age,
        height_cm=request.height_cm,
        weight_kg=request.weight_kg,
        fitness_goal=request.fitness_goal,
        target_calories_per_day=request.target_calories_per_day
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info(f"New user registered: {new_user.username} (ID: {new_user.id})")
    return new_user

@app.post("/login")
async def login(request: schemas.LoginUser, db: Session = Depends(get_db)):
    """User login endpoint."""
    user = db.query(models.User).filter(models.User.username == request.username).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # TODO: Use proper password hashing comparison (bcrypt)
    if user.password != request.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {
        "message": f"Welcome back, {user.username}!",
        "user_id": user.id,
        "login_time": datetime.utcnow().isoformat()
    }

@app.get("/user/{user_id}", response_model=schemas.UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user profile information."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# ============================================================================
# FOOD LOG ENDPOINTS WITH AI INTEGRATION
# ============================================================================

@app.post("/user/{user_id}/food", response_model=schemas.FoodLogResponse)
async def log_food(user_id: int, request: schemas.CreateFoodLog, db: Session = Depends(get_db)):
    """Create a new food log entry with AI embedding generation."""
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Create food log with vector integration
        new_food_log = await crud.create_food_log (db, user_id, request)
        logger.info(f"Food log created: {new_food_log.id} for user {user_id}")
        return new_food_log
        
    except ValueError as e:
        # Handle validation errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create food log: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create food log")

@app.get("/user/{user_id}/food", response_model=schemas.FoodLogListResponse)
async def get_food_logs(
    user_id: int,
    limit: int = Query(10, ge=1, le=100, description="Number of records to return"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    date_filter: Optional[date] = Query(None, description="Filter by specific date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get user's food logs with pagination and optional date filtering."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        food_logs, total_count = crud.get_food_logs(db, user_id, skip, limit, date_filter)
        return {
            "food_logs": food_logs,
            "total_count": total_count,
            "page_info": {
                "skip": skip,
                "limit": limit,
                "has_more": total_count > (skip + limit)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get food logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve food logs")

@app.get("/user/{user_id}/food/search")
async def search_similar_foods(
    user_id: int,
    query: str = Query(..., min_length=2, max_length=200, description="Food description to search for"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    global_search: bool = Query(False, description="Search all users (true) or just current user (false)"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score (0.0-1.0)"),
    db: Session = Depends(get_db)
):
    """Search for similar food logs using AI similarity search."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        # Determine search scope
        search_user_id = None if global_search else user_id
        
        # Perform similarity search
        vector_results = await vector_db.search_similar_foods(
            query_text=query,
            user_id=search_user_id,
            top_k=limit,
            similarity_threshold=min_score
        )
        
        # Transform vector results to API format
        formatted_results = []
        for result in vector_results:
            metadata = result.get('metadata', {})
            formatted_results.append({
                "id": result['food_log_id'],  # Use the extracted food_log_id
                "similarity_score": result['score'],
                "food_name": metadata.get('food_name'),
                "meal_type": metadata.get('meal_type'),
                "quantity_g": metadata.get('quantity_g'),
                "calories_total": metadata.get('calories_total'),
                # Add other fields as needed
            })

        return {
            "query": query,
            "results": formatted_results,
            "search_scope": "global" if global_search else "user", 
            "total_found": len(formatted_results),
            "min_score": min_score
        }
        
    except Exception as e:
        logger.error(f"Food search failed for query '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail="Search operation failed")

@app.delete("/user/{user_id}/food/{food_log_id}")
async def delete_food_log(user_id: int, food_log_id: int, db: Session = Depends(get_db)):
    """Delete a food log from both PostgreSQL and Pinecone."""
    try:
        success = await crud.delete_food_log (db, food_log_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Food log not found or unauthorized")
        
        logger.info(f"Food log {food_log_id} deleted for user {user_id}")
        return {"message": "Food log deleted successfully", "food_log_id": food_log_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete food log {food_log_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete food log")

# ============================================================================
# EXERCISE LOG ENDPOINTS WITH AI INTEGRATION
# ============================================================================

@app.post("/user/{user_id}/exercise", response_model=schemas.ExerciseLogResponse)
async def log_exercise(user_id: int, request: schemas.CreateExerciseLog, db: Session = Depends(get_db)):
    """Create a new exercise log with AI embedding generation."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        new_exercise_log = await crud.create_exercise_log(db, user_id, request)
        logger.info(f"Exercise log created: {new_exercise_log.id} for user {user_id}")
        return new_exercise_log
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create exercise log: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create exercise log")

@app.get("/user/{user_id}/exercise", response_model=schemas.ExerciseLogListResponse)
async def get_exercise_logs(
    user_id: int,
    limit: int = Query(10, ge=1, le=100, description="Number of records to return"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    date_filter: Optional[date] = Query(None, description="Filter by specific date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get user's exercise logs with pagination and optional date filtering."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        exercise_logs, total_count = crud.get_exercise_logs(db, user_id, skip, limit, date_filter)
        return {
            "exercise_logs": exercise_logs,
            "total_count": total_count,
            "page_info": {
                "skip": skip,
                "limit": limit,
                "has_more": total_count > (skip + limit)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get exercise logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve exercise logs")

@app.get("/user/{user_id}/exercise/search")
async def search_similar_exercises(
    user_id: int,
    query: str = Query(..., min_length=2, max_length=200, description="Exercise description to search for"),
    limit: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    global_search: bool = Query(False, description="Search all users (true) or just current user (false)"),
    min_score: float = Query(0.0, ge=0.0, le=1.0, description="Minimum similarity score"),
    db: Session = Depends(get_db)
):
    """Search for similar exercise logs using AI similarity search."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        search_user_id = None if global_search else user_id
        
        results = await crud.search_similar_exercise_logs(
            db=db,
            query=query,
            user_id=search_user_id,
            limit=limit,
            min_score=min_score
        )
        
        return {
            "query": query,
            "results": results,
            "search_scope": "global" if global_search else "user",
            "total_found": len(results),
            "min_score": min_score
        }
        
    except Exception as e:
        logger.error(f"Exercise search failed for query '{query}': {str(e)}")
        raise HTTPException(status_code=500, detail="Search operation failed")

@app.delete("/user/{user_id}/exercise/{exercise_log_id}")
async def delete_exercise_log(user_id: int, exercise_log_id: int, db: Session = Depends(get_db)):
    """Delete an exercise log from both PostgreSQL and Pinecone."""
    try:
        success = await crud.delete_exercise_log(db, exercise_log_id, user_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Exercise log not found or unauthorized")
        
        logger.info(f"Exercise log {exercise_log_id} deleted for user {user_id}")
        return {"message": "Exercise log deleted successfully", "exercise_log_id": exercise_log_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete exercise log {exercise_log_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete exercise log")

# ============================================================================
# AI & VECTOR DATABASE ADMINISTRATION
# ============================================================================

@app.get("/admin/vector-stats")
async def get_vector_database_stats():
    """Get Pinecone index statistics - admin only endpoint."""
    try:
        stats = await vector_db.get_index_stats()
        return {
            "pinecone_stats": stats,
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get vector stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve vector database statistics")

@app.get("/health/vector")
async def vector_health_check():
    """Health check endpoint for vector database and Hugging Face API."""
    try:
        health_status = await vector_db.health_check()
        
        if health_status.get("status") == "unhealthy":
            error_msg = health_status.get("error", "Unknown error")
            raise HTTPException(status_code=503, detail=f"Vector database unhealthy: {error_msg}")
        
        return health_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Vector database health check failed")

@app.post("/admin/vector/reindex")
async def reindex_embeddings(
    user_id: Optional[int] = Query(None, description="Reindex specific user (optional)"),
    db: Session = Depends(get_db)
):
    """Regenerate embeddings for all or specific user's data - admin endpoint."""
    try:
        result = await crud.reindex_user_embeddings(db, user_id)
        return {
            "message": "Reindexing completed",
            "processed": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Reindexing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Reindexing operation failed")

# ============================================================================
# DASHBOARD & ANALYTICS
# ============================================================================

@app.get("/user/{user_id}/dashboard")
async def get_user_dashboard(
    user_id: int, 
    target_date: Optional[date] = Query(None, description="Target date for dashboard data"),
    db: Session = Depends(get_db)
):
    """Get comprehensive user dashboard with AI insights."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if not target_date:
        target_date = date.today()
    
    try:
        # Get day's data
        food_logs, _ = crud.get_food_logs (db, user_id, 0, 100, target_date)
        exercise_logs, _ = crud.get_exercise_logs(db, user_id, 0, 100, target_date)
        
        # Calculate nutrition totals
        total_calories_consumed = sum(log.calories_total for log in food_logs)
        total_calories_burned = sum(log.calories_burned or 0 for log in exercise_logs)
        total_protein = sum(log.protein_g or 0 for log in food_logs)
        total_carbs = sum(log.carbs_g or 0 for log in food_logs)
        total_fat = sum(log.fat_g or 0 for log in food_logs)
        
        # AI insights and embedding coverage
        foods_with_embeddings = sum(1 for log in food_logs if log.has_embedding)
        exercises_with_embeddings = sum(1 for log in exercise_logs if log.has_embedding)
        
        # Calculate progress percentages
        calorie_progress = min(100, (total_calories_consumed / user.target_calories_per_day * 100)) if user.target_calories_per_day > 0 else 0
        
        return {
            "user_info": {
                "user_id": user_id,
                "username": user.username,
                "fitness_goal": user.fitness_goal
            },
            "date": target_date.isoformat(),
            "nutrition": {
                "calories_consumed": total_calories_consumed,
                "calories_burned": total_calories_burned,
                "net_calories": total_calories_consumed - total_calories_burned,
                "target_calories": user.target_calories_per_day,
                "calorie_progress_percent": round(calorie_progress, 1),
                "macros": {
                    "protein_g": round(total_protein, 1),
                    "carbs_g": round(total_carbs, 1),
                    "fat_g": round(total_fat, 1)
                }
            },
            "activity_summary": {
                "meals_logged": len(food_logs),
                "workouts_completed": len(exercise_logs),
                "total_exercise_duration_min": sum(log.duration_minutes or 0 for log in exercise_logs)
            },
            "ai_insights": {
                "embedding_coverage": {
                    "foods_with_ai": foods_with_embeddings,
                    "total_foods": len(food_logs),
                    "exercises_with_ai": exercises_with_embeddings,
                    "total_exercises": len(exercise_logs),
                    "ai_coverage_percent": round(
                        ((foods_with_embeddings + exercises_with_embeddings) / 
                         max(1, len(food_logs) + len(exercise_logs))) * 100, 1
                    )
                },
                "recommendations_available": foods_with_embeddings > 0 or exercises_with_embeddings > 0
            },
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dashboard generation failed for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate dashboard")

# ============================================================================
# ANALYTICS & TRENDS
# ============================================================================

@app.get("/user/{user_id}/analytics/weekly")
async def get_weekly_analytics(
    user_id: int,
    weeks_back: int = Query(4, ge=1, le=12, description="Number of weeks to analyze"),
    db: Session = Depends(get_db)
):
    """Get weekly trends and analytics for the user."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    try:
        end_date = date.today()
        start_date = end_date - timedelta(weeks=weeks_back)
        
        # Get all data in date range
        food_logs, _ = crud.get_exercise_logs(db, user_id, start_date, end_date)
        exercise_logs, _ = crud.get_exercise_logs(db, user_id, start_date, end_date)
        
        # Group by week and calculate averages
        weekly_stats = crud.calculate_weekly_trends(food_logs, exercise_logs, start_date, weeks_back)
        
        return {
            "user_id": user_id,
            "analysis_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "weeks_analyzed": weeks_back
            },
            "weekly_trends": weekly_stats,
            "summary": {
                "avg_daily_calories": round(sum(w["avg_calories"] for w in weekly_stats) / len(weekly_stats), 1),
                "avg_workouts_per_week": round(sum(w["total_workouts"] for w in weekly_stats) / len(weekly_stats), 1),
                "most_active_day": crud.get_most_active_day_of_week(exercise_logs)
            }
        }
        
    except Exception as e:
        logger.error(f"Weekly analytics failed for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate analytics")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)