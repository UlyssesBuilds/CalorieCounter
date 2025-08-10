#Models.py This is where we define our database models using SQLAlchemy
from sqlalchemy import Column, ForeignKey, Integer, String, Float, DateTime, Text, Boolean, JSON 
from .database import Base
from sqlalchemy.orm import relationship # allows FK relationships between tables
from sqlalchemy.sql import func # function for timestamps and counting records

#We are creating user table
class User(Base):
    __tablename__ = "users"

    # Core identification
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    username = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)  # TODO: Hash passwords before storing
    
    # Basic profile info
    first_name = Column(String, nullable=False)
    age = Column(Integer, nullable=True)
    
    # Essential fitness profile data
    height_cm = Column(Float, nullable=True)
    weight_kg = Column(Float, nullable=True)
    fitness_goal = Column(String, nullable=True)  # Weight Loss/Weight Gain/Muscle Gain/Maintenance
    target_calories_per_day = Column(Integer, nullable=True)
    
    # System fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    food_logs = relationship("FoodLog", back_populates="user", cascade="all, delete-orphan")
    exercise_logs = relationship("ExerciseLog", back_populates="user", cascade="all, delete-orphan")
    goal_tracking = relationship("GoalTracking", back_populates="user", cascade="all, delete-orphan")
    ai_chat_logs = relationship("AIChatLog", back_populates="user", cascade="all, delete-orphan")

class FoodLog(Base):
    __tablename__ = "food_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Basic food info
    food_name = Column(String, nullable=False)
    quantity_g = Column(Float, nullable=False)  # Amount in grams
    calories_total = Column(Float, nullable=False)  # Total calories consumed
    
    # Basic macros (optional for MVP)
    protein_g = Column(Float, nullable=True)
    carbs_g = Column(Float, nullable=True)
    fat_g = Column(Float, nullable=True)
    
    # Meal context
    meal_type = Column(String, nullable=True)  # breakfast/lunch/dinner/snack
    meal_date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # AI features
    photo_url = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    ai_analysis_result = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Vector tracking (but no actual vector storage)
    has_embedding = Column(Boolean, default=False)  # Track if vectors exist in Pinecone
    embedding_version = Column(String, nullable=True)  # Track embedding model version
    
    # Image storage
    image_url = Column(String, nullable=True)

    # System fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="food_logs")


class ExerciseLog(Base):
    __tablename__ = "exercise_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Basic exercise info
    exercise_name = Column(String, nullable=False)
    exercise_type = Column(String, nullable=False)  # cardio/strength/other
    duration_minutes = Column(Integer, nullable=False)
    calories_burned = Column(Float, nullable=True)
    
    # Simple intensity tracking
    intensity_level = Column(String, nullable=True)  # low/moderate/high
    
    # Timing
    exercise_date = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Optional notes
    notes = Column(Text, nullable=True)

    #  AI fields
    ai_analysis_result = Column(JSON, nullable=True)
    
    # Vector tracking
    has_embedding = Column(Boolean, default=False)
    embedding_version = Column(String, nullable=True)
    
    # System fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="exercise_logs")


class GoalTracking(Base):
    __tablename__ = "goal_tracking"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Simple goal definition
    goal_type = Column(String, nullable=False)  # weight_loss/muscle_gain/exercise_frequency
    goal_name = Column(String, nullable=False)
    target_value = Column(Float, nullable=False)
    current_value = Column(Float, default=0.0)
    unit = Column(String, nullable=False)  # kg/lbs/days/workouts
    
    # Simple timeline
    target_date = Column(DateTime(timezone=True), nullable=False)
    status = Column(String, default="active")  # active/completed/paused
    
    # System fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="goal_tracking")


class AIChatLog(Base):
    __tablename__ = "ai_chat_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Basic conversation tracking
    session_id = Column(String, nullable=False, index=True)
    message_type = Column(String, nullable=False)  # user_message/ai_response
    message_content = Column(Text, nullable=False)
    conversation_context = Column(JSON, nullable=True)  # Store conversation history

    # Simple categorization
    topic_category = Column(String, nullable=True)  # nutrition/exercise/general

    # AI metadata
    model_used = Column(String, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)

    # Vector tracking
    has_embedding = Column(Boolean, default=False)
    embedding_version = Column(String, nullable=True)

    # AI metadata
    model_used = Column(String, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # System fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="ai_chat_logs")

class AIAnalysisQueue(Base):
    __tablename__ = "ai_analysis_queue"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    # What's being analyzed
    analysis_type = Column(String, nullable=False)  # 'food_image', 'exercise_form', 'chat_response'
    reference_id = Column(Integer, nullable=True)   # ID of the food_log, exercise_log, etc.
    
    # Input data for AI
    input_data = Column(JSON, nullable=False)  # Image URLs, text, etc.
    
    # Processing status
    status = Column(String, default="pending")  # pending, processing, completed, failed
    priority = Column(Integer, default=5)       # 1-10, higher = more urgent
    
    # Results
    result_data = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    processing_attempts = Column(Integer, default=0)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    user = relationship("User")