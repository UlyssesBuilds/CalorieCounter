#Models.py This is where we define our database models using SQLAlchemy
from sqlalchemy import Column, ForeignKey, Integer, String, Float
from .database import Base

#We are creating user table
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True)
    username = Column(String, unique=True)
    password = Column(String, nullable=False)
    first_name = Column(String, nullable=True)
    age = Column(Integer, nullable=True)