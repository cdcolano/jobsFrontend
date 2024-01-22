from fastapi import FastAPI, APIRouter
import time
import os
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from fastapi import FastAPI, File, UploadFile
from typing import List
from schemas import Prenda, User, ComprasCreate, Compra
from fastapi.responses import RedirectResponse, HTMLResponse
import os
import fastapi
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request,Depends
from jose import jwt,JWTError
from fastapi.responses import HTMLResponse
import shutil
import requests
from databases import Database
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pymongo
from pymongo import MongoClient
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordBearer
import re
from nltk.stem import PorterStemmer
import numpy as np

response_404 = {404: {"description": "Item not found"}}
response_403= {403:{"description": "Error en el inicio de sesion"}}
response_401= {401:{"description": "No autorizado"}}
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30 minutes
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days
ALGORITHM = "HS256"
JWT_SECRET_KEY = "gfg_jwt_secret_key"

index = pymongo.MongoClient("mongodb+srv://deusto:deusto@cluster0.knpxqxl.mongodb.net/prendas?retryWrites=true&w=majority")
 
# Select your database
db = client["your_database"]

# Select your collection
collection = db["your_collection"]

# Load all documents from the collection into memory
all_documents = collection.find({})

origins = [
    "http://localhost:3000",
    "localhost:3000"
]





def tfidf(query, positional_index, stopwords):
    query=query.lower()
    pattern = re.compile(r'[^a-zA-Z" ]') 
    query=re.sub(pattern, ' ', query) 
    pattern = r'\b(?:' + '|'.join(stopwords) + r')\b'
        # Use a lambda function to lower the matched words
    query = re.sub(pattern, "", query)
    stemmer = PorterStemmer()
    tokens=[stemmer.stem(word) for word in query.split()]
    N = len(getAllDocs(positional_index))  # CHANGE ONCE DATABASE IS READY 
    Scores = {}
  #  Length = np.zeros(N) 
  #  query_tf = {}
    for token in tokens:
        #wt_q = (1 + np.log10(query_tf[token])) if query_tf[token] > 0 else 0
        postings=positional_index.get(token)
        if postings is not None:
            for doc_id, idx_term in postings['postings'].items():
                if doc_id not in Scores:
                    Scores[doc_id]=0
                w_ft=(1+np.log10(len(idx_term)))* np.log10(N/postings['df'])
                Scores[doc_id]+=w_ft#*wt_q
            #Length[doc_id] += w_ft ** 2
    #Length = np.sqrt(Length)
    #Length = np.where(Length == 0, 1e-10, Length)
    #Scores = Scores / Length
    keys = np.array(list(Scores.keys()))
    values = np.array(list(Scores.values()))
    sorted_indices = np.argsort(values)[::-1]
# Sorting
    
    sorted_keys = keys[sorted_indices]
    sorted_values = values[sorted_indices]
    sorted_values=np.round(sorted_values,4)
    return sorted_keys

#for i, b in tfidf("wink drink ink", positional_index, stopwords):
#    print(i,b)

app = FastAPI(title="Gateway", openapi_url="/openapi.json")


@app.on_event("startup")
async def startup():
    await database.connect()

@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()

api_router = APIRouter()
origins=["*"]
#def getPrenda():
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

async def search_jobs(query: str):
    # Run your TF-IDF program to get relevant job IDs
    job_ids = tfidf(query)

    # Now, query the database for these job IDs
    query = "SELECT * FROM Jobs WHERE ID IN :ids"
    return await database.fetch_all(query=query, values={"ids": tuple(job_ids)})


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="http://localhost:4000/clientes/signin" )


async def get_current_user(token: str= Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY)#, algorithms=[ALGORITHM])
        userId = payload.get("userId")
        if userId is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return userId



   



app.include_router(api_router)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="debug")
