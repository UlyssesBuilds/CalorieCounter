# Schemas.py We create validators of the client data; they must send me this; so diff schmas for diff operations
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# Register, Login, Profile
class CreateUser(BaseModel):
    email: str
    password: str
    username: str
    first_name: str
    age: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    fitness_goal: Optional[str] = None
    target_calories_per_day: Optional[int] = None

class LoginUser(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    first_name: str
    age: Optional[int]
    height_cm: Optional[float]
    weight_kg: Optional[float]
    fitness_goal: Optional[str]
    target_calories_per_day: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

class UpdateUser(BaseModel):
    first_name: Optional[str] = None
    age: Optional[int] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    fitness_goal: Optional[str] = None
    target_calories_per_day: Optional[int] = None

# Food Log Schemas
class CreateFoodLog(BaseModel):
    food_name: str
    quantity_g: float
    calories_total: float
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    meal_type: Optional[str] = None  # breakfast/lunch/dinner/snack
    meal_date: datetime
    photo_url: Optional[str] = None

class FoodLogResponse(BaseModel):
    id: int
    food_name: str
    quantity_g: float
    calories_total: float
    protein_g: Optional[float]
    carbs_g: Optional[float]
    fat_g: Optional[float]
    meal_type: Optional[str]
    meal_date: datetime
    photo_url: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

# Exercise Log Schemas
class CreateExerciseLog(BaseModel):
    exercise_name: str
    exercise_type: str  # cardio/strength/other
    duration_minutes: int
    calories_burned: Optional[float] = None
    intensity_level: Optional[str] = None  # low/moderate/high
    exercise_date: datetime
    notes: Optional[str] = None

class ExerciseLogResponse(BaseModel):
    id: int
    exercise_name: str
    exercise_type: str
    duration_minutes: int
    calories_burned: Optional[float]
    intensity_level: Optional[str]
    exercise_date: datetime
    notes: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True
    

# Goal Tracking Schemas
class CreateGoal(BaseModel):
    goal_type: str  # weight_loss/muscle_gain/exercise_frequency
    goal_name: str
    target_value: float
    unit: str  # kg/lbs/days/workouts
    target_date: datetime

class GoalResponse(BaseModel):
    id: int
    goal_type: str
    goal_name: str
    target_value: float
    current_value: float
    unit: str
    target_date: datetime
    status: str
    created_at: datetime

    class Config:
        from_attributes = True

class UpdateGoal(BaseModel):
    current_value: Optional[float] = None
    status: Optional[str] = None  # active/completed/paused

# AI Chat Log Schemas
class CreateChatMessage(BaseModel):
    session_id: str
    message_type: str  # user_message/ai_response
    message_content: str
    topic_category: Optional[str] = None  # nutrition/exercise/general

class ChatMessageResponse(BaseModel):
    id: int
    session_id: str
    message_type: str
    message_content: str
    topic_category: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

# Response schemas for lists
class FoodLogListResponse(BaseModel):
    food_logs: List[FoodLogResponse]
    total_count: int

class ExerciseLogListResponse(BaseModel):
    exercise_logs: List[ExerciseLogResponse]
    total_count: int

class GoalListResponse(BaseModel):
    goals: List[GoalResponse]
    total_count: int

class ChatHistoryResponse(BaseModel):
    messages: List[ChatMessageResponse]
    total_count: int