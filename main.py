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
from dateutil import parser
from datetime import datetime
import regex
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import nltk
from langdetect import detect
from langcodes import Language
import gc

nltk.download('stopwords')


hostname = 'ms0806.utah.cloudlab.us'
database = 'jobs_db'
username = 'root'
port = 5432
password = 'root'  # It's not recommended to hardcode passwords in your scripts

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

connection = psycopg2.connect(
        host=hostname,
        dbname=database,
        user=username,
        password=password,
        port=port
    )

query=f"""
       SELECT Count(*) FROM jobs;

    """

# Execute the query
cursor = connection.cursor()
cursor.execute(query)
N = cursor.fetchone()[0]  # Fetch the JSON result
print(N)

query=f"""
       SELECT json_agg(ID) FROM jobs;

    """

cursor.execute(query)
DOC_IDS= cursor.fetchone()[0] #DOC_IDS could be replace to ID2DATE.keys() if memory ussage is bad


query = """
SELECT json_object_agg(ID, date_posted) FROM jobs;
"""
cursor = connection.cursor()
cursor.execute(query)
ID2DATE = cursor.fetchone()[0] #store the ID to Date to consider dates in the retrieval
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
collection=db['INDEX']

# total_documents = collection.count_documents({})

# start_time_load=time.time()
# results=collection.find({})
# positional_index = {}
# for document in tqdm(results, total=total_documents):
#     positional_index[document['_id']] = document
# #positional_index = {document['_id']: document for document in results}
# end_time_load=time.time()

def getAllDocs(positional_index):
    all_doc_ids = set()
    for term_data in positional_index.values():
        all_doc_ids.update(term_data['posting_list'].keys())
    return all_doc_ids



CURRENT_RESULT=[]
compiled_patterns = {}
stemmers = {}
stemmed_languages = ["arabic","danish","dutch","english","finnish","french","german","hungarian","italian","norwegian","portuguese","romanian","russian","spanish","swedish"]
non_alpha_pattern = regex.compile(r'\p{P}') 

def desp_preprocessing(text):
    try:
        language_code = detect(text)
        language_full_form = Language.get(language_code).language_name().lower()
    except:
        language_full_form = 'unknown'

    text = non_alpha_pattern.sub(" ", text)

    if language_full_form in stemmed_languages:
        stopwords_by_language = set(stopwords.words(language_full_form))
        
        if language_full_form not in compiled_patterns:
            pattern = r'\b(?:' + '|'.join(map(re.escape, stopwords_by_language)) + r')\b'
            compiled_patterns[language_full_form] = re.compile(pattern)
        
        text = compiled_patterns[language_full_form].sub(' ', text)
        
        if language_full_form not in stemmers:
            stemmers[language_full_form] = SnowballStemmer(language_full_form)
        stemmer = stemmers[language_full_form]
        
        words = [stemmer.stem(token) for token in text.split()]
    else:
        words = text.split()
    
    return words

def add_dates_to_score(Scores, connection):
    id_array=list(Scores.keys())
    query = """
    SELECT id, date_posted
    FROM jobs
    WHERE id = ANY(%s);
    """

    cursor = connection.cursor()
    cursor.execute(query, (id_array,))
    results = cursor.fetchall()
    print(results)

parsed_cache = {'': datetime(year=1900, month=1, day=1)} #Cache for quixcker parsing critical for timesaving in queries

def parse_date(date_str, dayfirst=True):
    if date_str in parsed_cache:  # Return cached result if available
        return parsed_cache[date_str]
    try:
        # Parse the date with dayfirst option
        parsed_date = parser.parse(date_str, dayfirst=dayfirst)
        parsed_cache[date_str] = parsed_date  # Cache the result
        return parsed_date
    except ValueError:
        print(f"Could not parse date: {date_str}")
        return None

def optimized_tfidf(query, N_DOCS):
    tokens=desp_preprocessing(query)
    Scores = {}
    current_time=time.time()
    for token in tokens:
        postings= collection.find_one({"_id": token})
        postings = positional_index.get(token)
        if postings:
            df = postings['df']
            idf = np.log10(N_DOCS / df)
            for doc_id, idx_term in postings['posting_list'].items():
                parsed_date = parse_date(ID2DATE[doc_id], dayfirst=True) #This would work but extremelly inefficient
                date_factor=current_time - parsed_date
                days_diff = date_factor.days
                date_factor = 1 / (days_diff + 1)  # Adding 1 to avoid division by zero and ensure recent docs have higher factor
                tf = 1 + np.log10(len(idx_term))
                Scores[doc_id] = Scores.get(doc_id, 0) + tf * idf * date_factor


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

def perform_phrase_search(query):
   # tokens=pre_process(query)
    postings = []
    for token in query:
        # Fetch the document for the current token from MongoDB
        document = collection.find_one({"_id": token})
        if document:
            postings.append(document.get('posting_list', {}))
        else:
            # If the token is not found, append an empty postings list
            postings.append({})
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

def perform_proximity_search(tokens, PROX):
    #tokens=pre_process(tokens)
    
    postings = []
    for token in tokens:
        # Fetch the document for the current token from MongoDB
        document = collection.find_one({"_id": token})
        if document:
            postings.append(document.get('posting_list', {}))
        else:
            # If the token is not found, append an empty postings list
            postings.append({})
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

non_alpha_pattern_boolean = regex.compile(r'[^\w\s#"()]+')
logical_operators = {'AND', 'OR', 'NOT'}

def desp_preprocessing_boolean(text):
    try:
        language_code = detect(text)
        language_full_form = Language.get(language_code).language_name().lower()
    except:
        language_full_form = 'unknown'
    
    # Removing punctuation except for '#'
    text = non_alpha_pattern_boolean.sub(" ", text)

    if language_full_form in stemmed_languages:
        stopwords_by_language = set(stopwords.words(language_full_form))
        # Preparing regex pattern without altering 'AND', 'NOT', 'OR'
        if language_full_form not in compiled_patterns:
            # Exclude 'AND', 'NOT', 'OR' from being treated as stopwords
            pattern_stopwords = stopwords_by_language - {'AND', 'NOT', 'OR'}
            pattern = r'\b(?:' + '|'.join(map(re.escape, pattern_stopwords)) + r')\b'
            compiled_patterns[language_full_form] = re.compile(pattern)
        
        text = compiled_patterns[language_full_form].sub(' ', text)
        
        if language_full_form not in stemmers:
            stemmers[language_full_form] = SnowballStemmer(language_full_form)
        stemmer = stemmers[language_full_form]
        
        # Tokenizing while preserving 'AND', 'NOT', 'OR', and '#'
        tokens = re.findall(r'(\bAND\b|\bOR\b|\bNOT\b|\b\w+\b|[#"\(\)])', text)
        words = [token if token in logical_operators else stemmer.stem(token) for token in tokens]
    else:
        # For languages not supported for stemming, tokenize while preserving specific tokens
        words = re.findall(r'(\bAND\b|\bOR\b|\bNOT\b|\b\w+\b|[#"\(\)])', text)
    
    return words
#boolean search function
def boolean_search(query):
    ######## IMPORTANT#####
    #NOT REMOVING #,(,),""
    #CHECKING IF #WORK
    #COMPROBAR QUE EL TDIDF SE ORDENA DESCENDENTEMENTE Y ELIMINAR OR AND Y ETC DE LA REGEX
    
    tokens=desp_preprocessing_boolean(query)

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
                current_result&=perform_proximity_search(word_for_phrase, distance)
                word_for_phrase.clear()
                proximity=False
        elif token == '"':
            if (not phrase): #start of the phrase is detected
                phrase=True
            else: #end of the phrase is detected search is performed
                current_result&=perform_phrase_search(word_for_phrase)
                word_for_phrase.clear()
                phrase=False
        elif  (phrase or proximity):
                word_for_phrase.append(token) #if phrase or proximity search is being activated the words are added to the list until the end is detected
        elif (hashtag):
                distance=int(token)  #distance is set
                hashtag=False
        else:
            dict_word= collection.find_one({"_id": token})
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
        CURRENT_RESULT=boolean_search(query=query)
    else:
        CURRENT_RESULT=optimized_tfidf(query, N_DOCS)
    return await retrieve_jobs(1,N_PAGE)
# Example usage
# sorted_keys = tfidf("your query", positional_index, stopwords, N_DOCS)


#for i, b in tfidf("wink drink ink", positional_index, stopwords):
#    print(i,b)


def fetch_documents(indexes):
    indexes_str = ','.join([f"'{index}'" for index in indexes])
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
    return result

pagination_offset=0

import json
@app.get("/jobs/")
async def retrieve_jobs(page: int, number_per_page: int):
    start_index = (page - 1) * number_per_page
    end_index = start_index + number_per_page
    
    
    document_indexes_needed = CURRENT_RESULT[start_index:end_index]
    current_length=len(CURRENT_RESULT)
    documents_fetched = fetch_documents(document_indexes_needed)
    additional_docs_needed = number_per_page - len(documents_fetched)
    
    # If more documents are needed, fetch additional documents
    while additional_docs_needed > 0 and start_index<current_length:
        pagination_offset+=additional_docs_needed
        # Adjust start and end indexes to fetch more documents
        new_start_index = end_index
        new_end_index = new_start_index + additional_docs_needed
        
        additional_indexes_needed = CURRENT_RESULT[new_start_index:new_end_index]
 
        additional_documents = fetch_documents(additional_indexes_needed)
        
        documents_fetched.extend(additional_documents)
        additional_docs_needed = number_per_page - len(documents_fetched)
        
        # Update buffer size or break condition to avoid infinite loops if needed
        
    return documents_fetched

##################################################
#TODO verify new method
##################################################

# async def retrieve_jobs(page: int = Query(1, description="Page number of the results"),
#                         number_per_page: int = Query(10, description="Number of results per page")):
#     # Calculate start and end indexes for the slice of documents needed
#     start_index = (page - 1) * number_per_page+pagination_offset
#     end_index = start_index + number_per_page

#     # Get the document indexes for the current page
#     document_indexes_needed = CURRENT_RESULT[start_index:end_index]

#     # Convert the list of indexes to a format suitable for SQL query ('IN' clause)
#     # Ensure each index is treated as a string literal in the query
#     indexes_str = ','.join([f"'{index}'" for index in document_indexes_needed])
#     # query = f"""
#     #     SELECT json_agg(docs.* ORDER BY idx)
#     #     FROM (
#     #     SELECT j.*, idx
#     #     FROM unnest(ARRAY[{indexes_str}]) WITH ORDINALITY AS x(idx)
#     #     JOIN jobs j ON j.id = x.idx
#     #     ) AS docs;
#     #     """
#     query=f"""
#         SELECT json_agg(j ORDER BY j.idx)
#         FROM (
#             SELECT j.*, x.idx
#             FROM unnest(ARRAY[{indexes_str}]::text[]) WITH ORDINALITY AS x(id, idx)
#             JOIN jobs j ON j.id = x.id
#         ) AS j;

#     """

#     # Execute the query
#     cursor = connection.cursor()
#     cursor.execute(query)
#     result = cursor.fetchone()[0]  # Fetch the JSON result

#     cursor.close()

#     # `result` is already a JSON string; return it directly
#     return result


def test_jobs(page: int = Query(1, description="Page number of the results"),
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
    
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="debug")
    CURRENT_RESULT=DOC_IDS[:10000]
    print(test_jobs(2, 20))
        # SQL query to retrieve table schema information
    # SQL query to fetch IDs (assuming IDs are stored in a column named 'id' in the 'jobs' table)

   
