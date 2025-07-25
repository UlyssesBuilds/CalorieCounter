from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# class Item(BaseModel):
#     name: str
#     price: float
#     is_offer: Union[bool, None] = None


# @app.get("/")
# async def read_root():
#     return {"hello":"World"}

# @app.get("/items/{item_id}")
# async def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}

# @app.put("/items/{item_id}")
# async def update_item(item_id: int, item: Item):
#     return {"item_id": item.name, item_id: item_id}

#login
@app.get("/")
async def login():
    
    return {"hello":"me"}

#home
@app.get("/")
async def home():
    return {"hello":"me"}

#profile
@app.get("/")
async def profile():
    
    return {"hello":"me"}
