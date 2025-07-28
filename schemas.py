# Schemas.py We create OOP classes to represent the data we to check coming in and store in the database
from pydantic import BaseModel, EmailStr

# Register, Login, Profile
class User(BaseModel):
    email: str
    password: str
    username: str
    first_name: str
    age: int


# class Activity(BaseModel):
#     activity_type: str #change to a drop down list in frontend
#     duration: int # in minutes
#     # calories_burned: float  || I need to calc calories burned based on activity and duration



# class Food(BaseModel):
#     food_name: str
#     quantity: float # in grams
#     calories: float # calories per 100 grams
#     protein: float # grams per 100 grams
#     carbs: float # grams per 100 grams
#     fats: float # grams per 100 grams
#     timelogged: Union[str, None] = None  # timestamp of when the food was logged
#     meal_type: Union[str, None] = None  # e.g., breakfast, lunch, dinner, snack


# #Sleep log
# class Sleep(BaseModel):
#     sleep_duration: int  # in minutes
#     sleep_quality: str  # e.g., good, fair, poor
#     notes: Union[str, None] = None  # additional notes about the sleep