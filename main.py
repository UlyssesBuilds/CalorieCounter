#main.py
from typing import Union, Optional #optional allows for None values as default in the model
from fastapi import FastAPI, Depends, HTTPException, status
from . import schemas, models # from . = same directory, import schemas from schemas.py. now I access Class by schemas.User
from .database import engine, SessionLocal
from sqlalchemy.orm import Session
from datetime import datetime, date, timedelta
app = FastAPI()


models.Base.metadata.drop_all(engine)   # Deletes all tables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! only for now
# Create the database tables if they don't exist
models.Base.metadata.create_all(engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
#Add JWT

#root endpoint
@app.get("/")
async def readroot():
    return {"message": "Welcome to the Calorie Counter API! Visit /docs for API documentation."}



 # I am creating a new user with the User class from schemas.py from client input passed through parameters
@app.post("/register", response_model=schemas.UserResponse)
async def register(request: schemas.CreateUser, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(models.User).filter(
        (models.User.email == request.email) | (models.User.username == request.username)
    ).first()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Email or username already registered")
    
    new_user = models.User(
        email=request.email,
        username=request.username,
        password=request.password,  # TODO: Hash this password
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
    
    return new_user  # Returns UserResponse schema automatically

@app.post("/login")
async def login(request: schemas.LoginUser, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == request.username).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    if user.password != request.password:  # TODO: Use password hashing
        raise HTTPException(status_code=401, detail="Incorrect password")
    
    return {"message": f"Welcome {user.username}!", "user_id": user.id}

@app.get("/user/{user_id}", response_model=schemas.UserResponse)
async def get_user_profile(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.put("/user/{user_id}", response_model=schemas.UserResponse)
async def update_user_profile(user_id: int, request: schemas.UpdateUser, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Update only the fields that were provided
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    
    db.commit()
    db.refresh(user)
    return user

# ============================================================================
# FOOD LOG ENDPOINTS
# ============================================================================

@app.post("/user/{user_id}/food", response_model=schemas.FoodLogResponse)
async def log_food(user_id: int, request: schemas.CreateFoodLog, db: Session = Depends(get_db)):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_food_log = models.FoodLog(
        user_id=user_id,
        food_name=request.food_name,
        quantity_g=request.quantity_g,
        calories_total=request.calories_total,
        protein_g=request.protein_g,
        carbs_g=request.carbs_g,
        fat_g=request.fat_g,
        meal_type=request.meal_type,
        meal_date=request.meal_date,
        photo_url=request.photo_url
    )
    
    db.add(new_food_log)
    db.commit()
    db.refresh(new_food_log)
    return new_food_log

@app.get("/user/{user_id}/food", response_model=schemas.FoodLogListResponse)
async def get_food_logs(
    user_id: int, 
    limit: int = 10, 
    skip: int = 0,
    date_filter: Optional[date] = None,
    db: Session = Depends(get_db)
):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    query = db.query(models.FoodLog).filter(models.FoodLog.user_id == user_id)
    
    # Filter by date if provided
    if date_filter:
        query = query.filter(models.FoodLog.meal_date >= date_filter)
        query = query.filter(models.FoodLog.meal_date < date_filter + timedelta(days=1))
    
    # Get total count and paginated results
    total_count = query.count()
    food_logs = query.order_by(models.FoodLog.meal_date.desc()).offset(skip).limit(limit).all()
    
    return {
        "food_logs": food_logs,
        "total_count": total_count
    }

@app.delete("/user/{user_id}/food/{food_log_id}")
async def delete_food_log(user_id: int, food_log_id: int, db: Session = Depends(get_db)):
    food_log = db.query(models.FoodLog).filter(
        models.FoodLog.id == food_log_id,
        models.FoodLog.user_id == user_id
    ).first()
    
    if not food_log:
        raise HTTPException(status_code=404, detail="Food log not found")
    
    db.delete(food_log)
    db.commit()
    return {"message": "Food log deleted successfully"}

# ============================================================================
# EXERCISE LOG ENDPOINTS  
# ============================================================================

@app.post("/user/{user_id}/exercise", response_model=schemas.ExerciseLogResponse)
async def log_exercise(user_id: int, request: schemas.CreateExerciseLog, db: Session = Depends(get_db)):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_exercise_log = models.ExerciseLog(
        user_id=user_id,
        exercise_name=request.exercise_name,
        exercise_type=request.exercise_type,
        duration_minutes=request.duration_minutes,
        calories_burned=request.calories_burned,
        intensity_level=request.intensity_level,
        exercise_date=request.exercise_date,
        notes=request.notes
    )
    
    db.add(new_exercise_log)
    db.commit()
    db.refresh(new_exercise_log)
    return new_exercise_log

@app.get("/user/{user_id}/exercise", response_model=schemas.ExerciseLogListResponse)
async def get_exercise_logs(
    user_id: int,
    limit: int = 10,
    skip: int = 0,
    date_filter: Optional[date] = None,
    db: Session = Depends(get_db)
):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    query = db.query(models.ExerciseLog).filter(models.ExerciseLog.user_id == user_id)
    
    # Filter by date if provided
    if date_filter:
        query = query.filter(models.ExerciseLog.exercise_date >= date_filter)
        query = query.filter(models.ExerciseLog.exercise_date < date_filter + timedelta(days=1))
    
    total_count = query.count()
    exercise_logs = query.order_by(models.ExerciseLog.exercise_date.desc()).offset(skip).limit(limit).all()
    
    return {
        "exercise_logs": exercise_logs,
        "total_count": total_count
    }

@app.delete("/user/{user_id}/exercise/{exercise_log_id}")
async def delete_exercise_log(user_id: int, exercise_log_id: int, db: Session = Depends(get_db)):
    exercise_log = db.query(models.ExerciseLog).filter(
        models.ExerciseLog.id == exercise_log_id,
        models.ExerciseLog.user_id == user_id
    ).first()
    
    if not exercise_log:
        raise HTTPException(status_code=404, detail="Exercise log not found")
    
    db.delete(exercise_log)
    db.commit()
    return {"message": "Exercise log deleted successfully"}

# ============================================================================
# GOAL TRACKING ENDPOINTS
# ============================================================================

@app.post("/user/{user_id}/goals", response_model=schemas.GoalResponse)
async def create_goal(user_id: int, request: schemas.CreateGoal, db: Session = Depends(get_db)):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_goal = models.GoalTracking(
        user_id=user_id,
        goal_type=request.goal_type,
        goal_name=request.goal_name,
        target_value=request.target_value,
        unit=request.unit,
        target_date=request.target_date
    )
    
    db.add(new_goal)
    db.commit()
    db.refresh(new_goal)
    return new_goal

@app.get("/user/{user_id}/goals", response_model=schemas.GoalListResponse)
async def get_goals(user_id: int, db: Session = Depends(get_db)):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    goals = db.query(models.GoalTracking).filter(models.GoalTracking.user_id == user_id).all()
    
    return {
        "goals": goals,
        "total_count": len(goals)
    }

@app.put("/user/{user_id}/goals/{goal_id}", response_model=schemas.GoalResponse)
async def update_goal(user_id: int, goal_id: int, request: schemas.UpdateGoal, db: Session = Depends(get_db)):
    goal = db.query(models.GoalTracking).filter(
        models.GoalTracking.id == goal_id,
        models.GoalTracking.user_id == user_id
    ).first()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    # Update only provided fields
    update_data = request.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(goal, field, value)
    
    db.commit()
    db.refresh(goal)
    return goal

# ============================================================================
# AI CHAT LOG ENDPOINTS (Optional for MVP)
# ============================================================================

@app.post("/user/{user_id}/chat", response_model=schemas.ChatMessageResponse)
async def log_chat_message(user_id: int, request: schemas.CreateChatMessage, db: Session = Depends(get_db)):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    new_chat_log = models.AIChatLog(
        user_id=user_id,
        session_id=request.session_id,
        message_type=request.message_type,
        message_content=request.message_content,
        topic_category=request.topic_category
    )
    
    db.add(new_chat_log)
    db.commit()
    db.refresh(new_chat_log)
    return new_chat_log

@app.get("/user/{user_id}/chat/{session_id}", response_model=schemas.ChatHistoryResponse)
async def get_chat_history(user_id: int, session_id: str, db: Session = Depends(get_db)):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    messages = db.query(models.AIChatLog).filter(
        models.AIChatLog.user_id == user_id,
        models.AIChatLog.session_id == session_id
    ).order_by(models.AIChatLog.created_at.asc()).all()
    
    return {
        "messages": messages,
        "total_count": len(messages)
    }

# ============================================================================
# DASHBOARD/SUMMARY ENDPOINTS
# ============================================================================

@app.get("/user/{user_id}/dashboard")
async def get_user_dashboard(user_id: int, target_date: Optional[date] = None, db: Session = Depends(get_db)):
    # Verify user exists
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Default to today if no date provided
    if not target_date:
        target_date = date.today()
    
    # Get today's food logs
    food_logs = db.query(models.FoodLog).filter(
        models.FoodLog.user_id == user_id,
        models.FoodLog.meal_date >= target_date,
        models.FoodLog.meal_date < target_date + timedelta(days=1)
    ).all()
    
    # Get today's exercise logs
    exercise_logs = db.query(models.ExerciseLog).filter(
        models.ExerciseLog.user_id == user_id,
        models.ExerciseLog.exercise_date >= target_date,
        models.ExerciseLog.exercise_date < target_date + timedelta(days=1)
    ).all()
    
    # Calculate totals
    total_calories_consumed = sum(log.calories_total for log in food_logs)
    total_calories_burned = sum(log.calories_burned or 0 for log in exercise_logs)
    total_protein = sum(log.protein_g or 0 for log in food_logs)
    total_carbs = sum(log.carbs_g or 0 for log in food_logs)
    total_fat = sum(log.fat_g or 0 for log in food_logs)
    
    return {
        "user_id": user_id,
        "date": target_date,
        "calories_consumed": total_calories_consumed,
        "calories_burned": total_calories_burned,
        "net_calories": total_calories_consumed - total_calories_burned,
        "target_calories": user.target_calories_per_day,
        "macros": {
            "protein_g": total_protein,
            "carbs_g": total_carbs,
            "fat_g": total_fat
        },
        "meals_logged": len(food_logs),
        "workouts_completed": len(exercise_logs)
    }