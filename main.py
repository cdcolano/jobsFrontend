from fastapi import FastAPI, APIRouter
import time
import os
from fastapi import FastAPI, Body, HTTPException, status
from fastapi.responses import Response, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from fastapi import FastAPI, File, UploadFile
from typing import List
#from schemas import Prenda, User, ComprasCreate, Compra
from fastapi.responses import RedirectResponse, HTMLResponse
import os
import fastapi
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, Request,Depends, Query
#from jose import jwt,JWTError
from fastapi.responses import HTMLResponse
import shutil
import requests
#from databases import Database
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pymongo
from pymongo import MongoClient
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.security import OAuth2PasswordBearer
import re
import itertools
from nltk.stem import PorterStemmer
import numpy as np
from urllib.parse import quote_plus
import psycopg2
from tqdm import tqdm


response_404 = {404: {"description": "Item not found"}}
response_403= {403:{"description": "Error en el inicio de sesion"}}
response_401= {401:{"description": "No autorizado"}}
ACCESS_TOKEN_EXPIRE_MINUTES = 30  # 30 minutes
REFRESH_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # 7 days
ALGORITHM = "HS256"
JWT_SECRET_KEY = "gfg_jwt_secret_key"

#db_connection = pymongo.MongoClient("mongodb+srv://deusto:deusto@cluster0.knpxqxl.mongodb.net/prendas?retryWrites=true&w=majority")
 
# Select your database
#db = db_connection["your_database"]

# Select your collection
#collection = db["your_collection"]

# Load all documents from the collection into memory
#positional_index = collection.find({})

N_DOCS=0

connection = psycopg2.connect(user="postgres",
                                  password="pass",
                                  host="127.0.0.1",
                                  port="5432",
                                  database="JOBS")

query=f"""
       SELECT Count(*) FROM jobs;

    """

# Execute the query
cursor = connection.cursor()
cursor.execute(query)
N = cursor.fetchone()[0]  # Fetch the JSON result

query=f"""
       SELECT ID FROM jobs;

    """

cursor.execute(query)
DOC_IDS= cursor.fetchall()

cursor.close()

origins = [
    "http://localhost:3000",
    "localhost:3000"
]

app = FastAPI(title="Gateway", openapi_url="/openapi.json")



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

USERNAME= quote_plus('ttds')
PASSWORD = quote_plus('ttds')

uri = 'mongodb+srv://' + USERNAME + ':' + PASSWORD + "@ttds-cluster.vubotvd.mongodb.net/?retryWrites=true&w=majority"
# Create a new client and connect to the server
client = MongoClient(uri)


db=client.INDEX_DB

total_documents = db['INDEX'].count_documents({})

start_time_load=time.time()
results=db['INDEX'].find({})
positional_index = {}
for document in tqdm(results, total=total_documents):
    positional_index[document['_id']] = document
#positional_index = {document['_id']: document for document in results}
end_time_load=time.time()

def getAllDocs(positional_index):
    all_doc_ids = set()
    for term_data in positional_index.values():
        all_doc_ids.update(term_data['posting_list'].keys())
    return all_doc_ids






with open('./stopwords.txt', "r+",encoding='utf-8-sig') as file:
    stopwords=file.read()


CURRENT_RESULT=[]

pattern_non_alpha = re.compile(r'[^a-zA-Z" ]')
pattern_stopwords = re.compile(r'\b(?:' + '|'.join(stopwords) + r')\b')

def pre_process(query):
    query = pattern_non_alpha.sub(' ', query.lower())
    query = pattern_stopwords.sub(" ", query)

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in query.split()]
    return tokens

def optimized_tfidf(query, positional_index, stopwords, N_DOCS):
    tokens=pre_process(query)
    Scores = {}
    for token in tokens:
        postings = positional_index.get(token)
        if postings:
            df = postings['df']
            idf = np.log10(N_DOCS / df)
            for doc_id, idx_term in postings['posting_list'].items():
                tf = 1 + np.log10(len(idx_term))
                Scores[doc_id] = Scores.get(doc_id, 0) + tf * idf

    #sorted_docs = sorted(Scores.items(), key=lambda x: x[1], reverse=True)
    sorted_docs= sorted(Scores, key=Scores.get, reverse=True)
    #print(sorted_docs)
    return  sorted_docs

    #return [(doc_id, round(score, 4)) for doc_id, score in sorted_docs]




from multiprocessing import Pool


def process_token(args):
    token, positional_index, N_DOCS = args
    Scores = {}
    postings = positional_index.get(token)
    if postings:
        df = postings['df']
        idf = np.log10(N_DOCS / df)
        for doc_id, idx_term in postings['posting_list'].items():
            tf = 1 + np.log10(len(idx_term))
            Scores[doc_id] = tf * idf
    return Scores




def tfidf(query, positional_index, stopwords, N_DOCS):
    query = re.sub(r'[^a-zA-Z" ]', ' ', query.lower())
    query = re.sub(r'\b(?:' + '|'.join(stopwords) + r')\b', " ", query)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in query.split()]

    with Pool(processes=4) as pool:
        results = pool.map(process_token, [(token, positional_index, N_DOCS) for token in tokens])

    # Combine scores from each token
    combined_scores = {}
    for score_dict in results:
        for doc_id, score in score_dict.items():
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + score

    # Sort and return results
    sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs

def perform_phrase_search(query, positional_index):
   # tokens=pre_process(query)
    postings = [positional_index.get(token, {"posting_list":{}})['posting_list'] for token in query] #get postings for tokens
    # Display Result
    common_doc_ids = set.intersection(*(set(posting.keys()) for posting in postings)) #perform intersection to get common docs

    final_doc_ids = set()
    for doc_id in common_doc_ids:
        positions = [posting[doc_id] for posting in postings]
        combinations=list(itertools.product(*positions)) #calculate all combinations of positions of different tokens in a doc
        for combination in combinations:
            valid=all([combination[i]+1==combination[i+1] for i in range(len(combination)-1)]) #checking if one token is followed by the other
            if(valid):
                final_doc_ids.add(doc_id)
                break
    return final_doc_ids

def perform_proximity_search(tokens, PROX, positional_index):
    #tokens=pre_process(tokens)
    postings = [positional_index.get(token, {"posting_list":{}})['posting_list'] for token in tokens]#get postings for tokens
    # Display Result
    common_doc_ids = set.intersection(*(set(posting.keys()) for posting in postings)) #perform intersection to get common docs

    final_doc_ids = set()
    for doc_id in common_doc_ids:
        positions = [posting[doc_id] for posting in postings]
        combinations=list(itertools.product(*positions))#calculate all combinations of positions of different tokens in a doc
        for combination in combinations:
            valid=all([abs(combination[i]-combination[i+1])<=PROX for i in range(len(combination)-1)])#checking if the distance is lower than the max distance
            if(valid):
                final_doc_ids.add(doc_id)
                break
    return final_doc_ids


#boolean search function
def boolean_search(positional_index, query, stopwords):
    ######## IMPORTANT#####
    #NOT REMOVING #,(,),""
    #CHECKING IF #WORK
    #COMPROBAR QUE EL TDIDF SE ORDENA DESCENDENTEMENTE Y ELIMINAR OR AND Y ETC DE LA REGEX
    
    pattern = r'\b(?!AND\b|OR\b|NOT\b)\w+\b'
        # Use a lambda function to lower the matched words
    query = re.sub(pattern, lambda x: x.group().lower(), query)  #query to lowercase except AND OR NOT
    pattern = re.compile(r'[^a-zA-Z0-9#()" ]') 
    query=re.sub(pattern, ' ', query) 
    pattern = r'\b(?!AND\b|OR\b|NOT\b)(' + '|'.join(stopwords) + r')\b' #removing stopwords from query except AND OR NOT 
    query = re.sub(pattern, " ", query)
    stemmer = PorterStemmer()
    pattern = r'\b(?!AND\b|OR\b|NOT\b)\w+\b'
    query = re.sub(pattern, lambda x: stemmer.stem(x.group()), query) #performing stemming except AND OR NOT
    tokens= re.findall(r'(\bAND\b|\bOR\b|\bNOT\b|\b\w+\b|["#\(\)])',query)#splitting on spaces and specific operators

    #doc_ids=set(getAllDocs(positional_index)) #if query is empty all docs are retrived
    current_result=DOC_IDS.copy()
    operators=[]
    word_for_phrase = []
    phrase=False
    distance=0
    proximity=False
    hashtag=False
    #boolean search
    #Dificulty while separating operators (boolean, proximity)

    for token in tokens:
        if token in ["AND", "OR", "NOT"]: 
            operators.append(token)#appended to the stack of operators
        elif token == "#": #next token should be maximum distance of proximity search
            hashtag=True
        elif token == "(":
                proximity=True #next tokens should be added to a list and perform proximity search over that list
        elif token ==")":#proximity search should be performed
                current_result&=perform_proximity_search(word_for_phrase, distance, positional_index)
                word_for_phrase.clear()
                proximity=False
        elif token == '"':
            if (not phrase): #start of the phrase is detected
                phrase=True
            else: #end of the phrase is detected search is performed
                current_result&=perform_phrase_search(word_for_phrase, positional_index)
                word_for_phrase.clear()
                phrase=False
        elif  (phrase or proximity):
                word_for_phrase.append(token) #if phrase or proximity search is being activated the words are added to the list until the end is detected
        elif (hashtag):
                distance=int(token)  #distance is set
                hashtag=False
        else:
            dict_word=positional_index.get(token) 
            if dict_word is not None: #word exist in the postings
                term_postings = set(dict_word['posting_list'].keys())
            else:
                term_postings=set()
            if (len(operators)==0):
                current_result&=term_postings #This is made for the first word
            while (len(operators)>0):
                operator=operators.pop()
                # perform operations in order of popping
                if operator == "AND":
                    current_result &=term_postings
                elif operator == "OR":
                    current_result |= term_postings
                elif operator == "NOT":
                    term_postings=DOC_IDS-term_postings
    return current_result


#CURRENT RESULT IS DESIGNED FOR PAGINATION, then establishing a page size is easey to keep track of the page and retrieve the actual docs
@app.get("/search/")
async def route_query(query: str, N_PAGE: int = Query(30, alias="page")):
    pattern = r'\b(AND|OR|NOT)\b|["#]'
    if re.search(pattern, query):
        CURRENT_RESULT=boolean_search(positional_index=positional_index, query=query, stopwords=stopwords)
    else:
        CURRENT_RESULT=optimized_tfidf(query, positional_index, stopwords, N_DOCS)
    retrieve_jobs(1,N_PAGE)
    return CURRENT_RESULT

# Example usage
# sorted_keys = tfidf("your query", positional_index, stopwords, N_DOCS)


#for i, b in tfidf("wink drink ink", positional_index, stopwords):
#    print(i,b)




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


import json
@app.get("/jobs/")
async def retrieve_jobs(page: int = Query(1, description="Page number of the results"),
                        number_per_page: int = Query(10, description="Number of results per page")):
    # Calculate start and end indexes for the slice of documents needed
    start_index = (page - 1) * number_per_page
    end_index = start_index + number_per_page

    # Get the document indexes for the current page
    document_indexes_needed = CURRENT_RESULT[start_index:end_index]

    # Convert the list of indexes to a format suitable for SQL query ('IN' clause)
    # Ensure each index is treated as a string literal in the query
    indexes_str = ','.join([f"'{index}'" for index in document_indexes_needed])
    # query = f"""
    #     SELECT json_agg(docs.* ORDER BY idx)
    #     FROM (
    #     SELECT j.*, idx
    #     FROM unnest(ARRAY[{indexes_str}]) WITH ORDINALITY AS x(idx)
    #     JOIN jobs j ON j.id = x.idx
    #     ) AS docs;
    #     """
    query=f"""
        SELECT json_agg(j ORDER BY j.idx)
        FROM (
            SELECT j.*, x.idx
            FROM unnest(ARRAY[{indexes_str}]::text[]) WITH ORDINALITY AS x(id, idx)
            JOIN jobs j ON j.id = x.id
        ) AS j;

    """

    # Execute the query
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()[0]  # Fetch the JSON result

    cursor.close()

    # `result` is already a JSON string; return it directly
    return result


    



   



app.include_router(api_router)


if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn
    #print(positional_index)
    query='#15(dow,stocks)'
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="debug")
    start=time.time()
   
    jobs=["ea1jobs-119731924",
"ea1jobs-119768764",
"ea1jobs-119732017",
"ea1jobs-119768865",
"ea1jobs-119768868",
"ea1jobs-119768922",
"ea1jobs-119768984",
"ea1jobs-119768985",
"ea1jobs-119809567",
"ea1jobs-119809571",
"ea1jobs-119809647",
"ea1jobs-119809653",
"ea1jobs-119809654",
"ea1jobs-119809655",
"ea1jobs-119809670",
"ea1jobs-119809860",
"ea1jobs-119809865",
"ea1jobs-119809870",
"ea1jobs-119850379",
"ea1jobs-119850500",
"ea1jobs-119850519",
"ea1jobs-119850545",
"ea1jobs-119890143",
"ea1jobs-119890150",
"ea1jobs-119890163",
"ea1jobs-119890166",
"ea1jobs-119890167",
"ea1jobs-119890192",
"ea1jobs-119890195",
"ea1jobs-119890217"]
    result=retrieve_jobs(jobs,1,20,connection)
    #results1=route_query(query)
    end=time.time()
    print(end-start)
    print(result[0])
    print("TIEMPO DE CARGA")
    print(end_time_load-start_time_load)
    #print(results1)

