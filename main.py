from typing import Union, Optional #optional allows for None values as default in the model
from fastapi import FastAPI, Depends, HTTPException, status
from . import schemas, models # from . = same directory, import schemas from schemas.py. now I access Class by schemas.User
from .database import engine, SessionLocal
from sqlalchemy.orm import Session
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

#login
@app.get("/")
async def readroot():
    return {"message": "Welcome to the Calorie Counter API! Visit /docs for API documentation."}


 # I am creating a new user with the User class from schemas.py from client input passed through parameters
@app.post("/register")
async def register(request: schemas.User, db: Session = Depends(get_db)):
    new_user = models.User(
        email=request.email, 
        username=request.username, 
        password=request.password,
        first_name=request.first_name, 
        age=request.age)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)  # This gets the generated ID
    return {'data': f'{request.username} has been registered successfully!'}


# @app.post("/login")
# async def login():
#     return {"hello":"world"}

# @app.get("/profile")
# async def get_profile():
#     return {"hello":"world"}

# @app.post("/profile")
# async def update_profile():
#     return {"hello":"world"}


# # Home webpage Route will allow for someone to log calories and food
# # you can add, get, and delete food logs. 
# # add a get all food / activity and daily food / activty (pagination returns)

# #activity log


# @app.post("/activity")
# async def log_activty():
#     return {"data":"{'name': 'John',}"}


# @app.get("/activity?limit=10") #added limit for pagination
# async def get_activty_log():
#     # only get 
#     return {'data':"world"}

# @app.delete("/activity")
# async def delete_activity_log():
#     return {"hello":"world"}



# @app.post("/food")
# async def log_food():    
#     return {"hello":"world"}

# @app.get("/food")
# async def get_food_log(): 
#     return {"hello":"world"}

# @app.delete("/food")
# async def delete_food_log():
#     return {"hello":"world"}



# @app.post("/sleep")
# async def log_sleep():
#     return {"hello":"world"}
# @app.get("/sleep")
# async def get_sleep_log():
#     return {"hello":"world"}
# @app.delete("/sleep")
# async def delete_sleep_log():
#     return {"hello":"world"}