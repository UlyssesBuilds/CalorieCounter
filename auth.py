import os
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import jwt
from datetime import datetime, timedelta
from . import models, schemas
from typing import Optional
from dotenv import load_dotenv
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .database import get_db


load_dotenv()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT config
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

# Security scheme for dependency injection
security = HTTPBearer()

def hash_password(password: str) -> str:
    """Hash a plain password."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_user(
    db: Session,
    username: str,
    email: str,
    password: str,
    first_name: Optional[str] = None,
    age: Optional[int] = None,
    height_cm: Optional[float] = None,
    weight_kg: Optional[float] = None,
    fitness_goal: Optional[str] = None,
    target_calories_per_day: Optional[float] = None
) -> models.User:
    """Create a new user with hashed password."""
    hashed_password = hash_password(password)
    new_user = models.User(
        username=username,
        email=email,
        password=hashed_password,
        first_name=first_name,
        age=age,
        height_cm=height_cm,
        weight_kg=weight_kg,
        fitness_goal=fitness_goal,
        target_calories_per_day=target_calories_per_day
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

def authenticate_user(db: Session, username: str, password: str) -> Optional[models.User]:
    """Authenticate a user by username and password."""
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        return None
    if not verify_password(password, user.password):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)  # You'll need to import get_db from your database module
) -> models.User:
    """Dependency to get the current authenticated user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    payload = verify_token(credentials.credentials)
    if payload is None:
        raise credentials_exception
    
    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception
    
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise credentials_exception
    
    return user

# Authentication endpoints - move these from main.py
def register_user(request: schemas.CreateUser, db: Session) -> models.User:
    """Register a new user - business logic for registration."""
    # Check for existing user
    existing_user = db.query(models.User).filter(
        (models.User.email == request.email) | (models.User.username == request.username)
    ).first()
    
    if existing_user:
        detail = "Email already registered" if existing_user.email == request.email else "Username already taken"
        raise HTTPException(status_code=400, detail=detail)
    
    # Create new user with hashed password
    new_user = create_user(
        db=db,
        username=request.username,
        email=request.email,
        password=request.password,
        first_name=request.first_name,
        age=request.age,
        height_cm=request.height_cm,
        weight_kg=request.weight_kg,
        fitness_goal=request.fitness_goal,
        target_calories_per_day=request.target_calories_per_day
    )
    
    return new_user

def login_user(request: schemas.LoginUser, db: Session) -> dict:
    """Login user - business logic for authentication."""
    user = authenticate_user(db, request.username, request.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, 
        expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username,
        "message": f"Welcome back, {user.username}!"
    }

# Legacy function for backward compatibility (if still used elsewhere)
def verify_user(db: Session, username: str, password: str) -> bool:
    """Legacy function - use authenticate_user instead."""
    user = authenticate_user(db, username, password)
    return user is not None