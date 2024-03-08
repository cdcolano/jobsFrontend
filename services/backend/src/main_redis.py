import time
import os
import itertools
import numpy as np
import psycopg2
import regex
import re
import nltk
import redis
import uvicorn
import threading
import schedule
import logging

import config.pg_config as db
import searchService

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from tqdm import tqdm
from dateutil import parser
from datetime import datetime
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect
from langcodes import Language
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from thesaurus import job_posting_thesaurus
from utils.preprocessing import preprocess,  preprocess_query, BOOL_OPERATORS


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('uvicorn')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
async def startup_event():
    database_instance = db.Database(**PG_CONNECTION_CONFIG)
    await database_instance.connect()
    app.state.db = database_instance
    logger.info("Server Startup")


@app.on_event("shutdown")
async def shutdown_event():
    if not app.state.db:
        await app.state.db.close()
    logger.info("Server Shutdown")


def parse_date(date_str, current_time, dayfirst=True):
    if date_str in parsed_cache:  # Return cached result if available
        return parsed_cache[date_str]
    try:
        # Parse the date with dayfirst option
        parsed_date = parser.parse(date_str, dayfirst=dayfirst)
        date_factor = current_time - parsed_date
        days_diff = abs(date_factor.days)
        date_factor = 1 / (
                1 + days_diff / 30)  # Adding 1 to avoid division by zero and ensure recent docs have higher factor
        parsed_cache[date_str] = date_factor  # Cache the result
        return date_factor
    except ValueError:
        print(f"Could not parse date: {date_str}")
        return None


def fetch_token(token):
    return r.hgetall(token)


def fetch_postings(query):
    with ThreadPoolExecutor(max_workers=10) as executor:
        postings_list = list(executor.map(fetch_token, query))
    return postings_list


def update_database_info():
    global DOC_IDS, N, ID2DATE
    query = f"""
        SELECT Count(*) FROM jobs;

        """

    cursor = connection.cursor()
    cursor.execute(query)
    N = cursor.fetchone()[0]  # Fetch the JSON result

    query = f"""
        SELECT json_agg(ID::INTEGER) FROM jobs;

        """

    cursor.execute(query)
    DOC_IDS = cursor.fetchone()[0]  # DOC_IDS could be replace to ID2DATE.keys() if memory ussage is bad

    query = """
    SELECT json_object_agg(ID, date_posted) FROM jobs;
    """
    cursor = connection.cursor()
    cursor.execute(query)
    current_time = datetime.now()
    ID2DATE = cursor.fetchone()[0]  # store the ID to Date to consider dates in the retrieval
    for key, value in tqdm(ID2DATE.items(), desc="Parsing dates"):
        ID2DATE[key] = parse_date(value, current_time)
    cursor.close()
    ID2DATE[''] = parse_date("", current_time)


def getAllDocs(positional_index):
    all_doc_ids = set()
    for term_data in positional_index.values():
        all_doc_ids.update(term_data['posting_list'].keys())
    return all_doc_ids


def desp_preprocessing(text):
    try:
        language_code = detect(text)
        language_full_form = Language.get(language_code).language_name().lower()
    except:
        language_full_form = 'unknown'

    text = non_alpha_pattern.sub(" ", text)
    text = text.lower()
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
    id_array = list(Scores.keys())
    query = """
    SELECT id, date_posted
    FROM jobs
    WHERE id = ANY(%s);
    """

    cursor = connection.cursor()
    cursor.execute(query, (id_array,))
    results = cursor.fetchall()


def optimized_tfidf(query, N_DOCS):
    tokens = preprocess(query)
    Scores = {}
    # Use pipeline to reduce the number of calls to Redis
    postings_list = fetch_postings(tokens)
    for postings in postings_list:
        if postings:
            df = len(postings)
            idf = np.log10(N_DOCS / df)
            for doc_id, idx_term in postings.items():
                idx_term_count = len(idx_term.split(","))
                date_factor = ID2DATE.get(doc_id, ID2DATE[''])
                tf = 1 + np.log10(idx_term_count)
                doc_id = int(doc_id)
                Scores[doc_id] = Scores.get(doc_id, 0) + tf * idf * date_factor
    sorted_docs = sorted(Scores, key=Scores.get, reverse=True)
    return sorted_docs

    # return [(doc_id, round(score, 4)) for doc_id, score in sorted_docs]


def perform_phrase_search(query):
    # tokens=pre_process(query)
    postings = fetch_postings(query)
    # Display Result
    common_doc_ids = set.intersection(
        *(set(posting.keys()) for posting in postings))  # perform intersection to get common docs
    final_doc_ids = set()
    if len(postings) > 1:
        for doc_id in common_doc_ids:
            positions = list(np.array([posting[doc_id].split(",") for posting in postings], dtype=int))
            combinations = list(
                itertools.product(*positions))  # calculate all combinations of positions of different tokens in a doc
            for combination in combinations:
                valid = all([combination[i] + 1 == combination[i + 1] for i in
                             range(len(combination) - 1)])  # checking if one token is followed by the other
                if (valid):
                    final_doc_ids.add(int(doc_id))
                    break
        return final_doc_ids
    else:
        common_doc_ids = set(list(np.array(list(common_doc_ids), dtype=int)))
        return common_doc_ids


def perform_proximity_search(tokens, PROX):
    # tokens=pre_process(tokens)
    postings = fetch_postings(query)
    # Display Result
    common_doc_ids = set.intersection(
        *(list(posting.keys()) for posting in postings))  # perform intersection to get common docs
    final_doc_ids = set()
    if len(postings) > 1:
        for doc_id in common_doc_ids:
            positions = set(np.array([posting[doc_id].split(",") for posting in postings], dtype=int))
            combinations = list(
                itertools.product(*positions))  # calculate all combinations of positions of different tokens in a doc
            for combination in combinations:
                valid = all([abs(combination[i] - combination[i + 1]) <= PROX for i in
                             range(len(combination) - 1)])  # checking if the distance is lower than the max distance
                if (valid):
                    final_doc_ids.add(int(doc_id))
                    break
        return final_doc_ids
    else:
        common_doc_ids = set(list(np.array(list(common_doc_ids), dtype=int)))
        return common_doc_ids


def desp_preprocessing_boolean(text):
    try:
        language_code = detect(text)
        language_full_form = Language.get(language_code).language_name().lower()
    except:
        language_full_form = 'unknown'

    # Removing punctuation except for '#'
    text = non_alpha_pattern_boolean.sub(" ", text)
    pattern = r'\b(?!AND\b|OR\b|NOT\b)\w+\b'
    # Use a lambda function to lower the matched words
    text = re.sub(pattern, lambda x: x.group().lower(), text)
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


# boolean search function
def boolean_search(tokens):
    ######## IMPORTANT#####
    # NOT REMOVING #,(,),""
    # CHECKING IF #WORK
    # COMPROBAR QUE EL TDIDF SE ORDENA DESCENDENTEMENTE Y ELIMINAR OR AND Y ETC DE LA REGEX

    tokens = preprocess(tokens, tokenization_regex=tokenization_regex_boolean)
    # doc_ids=set(getAllDocs(positional_index)) #if query is empty all docs are retrived
    current_result = set(DOC_IDS)
    operators = []
    word_for_phrase = []
    phrase = False
    distance = 0
    proximity = False
    hashtag = False
    # boolean search
    # Dificulty while separating operators (boolean, proximity)

    for token in tokens:
        if token in ["AND", "OR", "NOT"]:
            operators.append(token)  # appended to the stack of operators
        elif token == "#":  # next token should be maximum distance of proximity search
            hashtag = True
        elif token == "(":
            proximity = True  # next tokens should be added to a list and perform proximity search over that list
        elif token == ")":  # proximity search should be performed
            current_result &= perform_proximity_search(word_for_phrase, distance)
            word_for_phrase.clear()
            proximity = False
        elif token == '"':
            if (not phrase):  # start of the phrase is detected
                phrase = True
            else:  # end of the phrase is detected search is performed
                current_result &= perform_phrase_search(word_for_phrase)
                word_for_phrase.clear()
                phrase = False
        elif (phrase or proximity):
            word_for_phrase.append(
                token)  # if phrase or proximity search is being activated the words are added to the list until the end is detected
        elif (hashtag):
            distance = int(token)  # distance is set
            hashtag = False
        else:
            postings = r.hgetall(token)
            # dict_word=positional_index.get(token)
            if postings is not None:  # word exist in the postings
                posting_numeric = np.array(list(postings.keys()), dtype=int)
                term_postings = set(list(posting_numeric))
            else:
                term_postings = set()
            if (len(operators) == 0):
                current_result &= term_postings  # This is made for the first word
            while (len(operators) > 0):
                operator = operators.pop()
                # perform operations in order of popping
                if operator == "AND":
                    current_result &= term_postings
                elif operator == "OR":
                    current_result |= term_postings
                elif operator == "NOT":
                    term_postings = DOC_IDS - term_postings
    return list(current_result)


def expand_query(query):
    new_query = query
    for word in query.split():
        if word in job_posting_thesaurus:
            for synonym in job_posting_thesaurus[word]:
                new_query = new_query + " " + synonym
    return new_query


@app.get("/search/")
async def search(query: str, request: Request, page: int = 1):
    _start_time = time.time()
    if page < 1:
        raise HTTPException(status_code=400, detail='Page value must be equal or greater than 1.')
    try:
        results = await searchService.search(query, request, page=page)
        return {
            'processing': time.time() - _start_time,
            'data': [
                query,
                [{k: v for k, v in x.items()} for x in results]
            ]
        }
    except Exception as e:  # General / Unknown error
        raise HTTPException(status_code=500, detail={'query': query, 'message': f'Internal server error. {e}'})


def fetch_documents(indexes):
    indexes_str = ','.join([f"'{index}'" for index in indexes])
    query = f"""
        SELECT json_agg(j ORDER BY j.idx)
        FROM (
            SELECT j.*, x.idx
            FROM unnest(ARRAY[{indexes_str}]::int[]) WITH ORDINALITY AS x(id, idx)
            JOIN jobs j ON j.id = x.id
        ) AS j;

    """

    # Execute the query
    cursor = connection.cursor()
    cursor.execute(query)
    result = cursor.fetchone()[0]  # Fetch the JSON result

    cursor.close()
    return result


@app.get("/jobs/")
async def retrieve_jobs(page: int, number_per_page: int):
    global pagination_offset
    start_index = (page - 1) * number_per_page
    end_index = start_index + number_per_page

    document_indexes_needed = CURRENT_RESULT[start_index:end_index]
    current_length = len(CURRENT_RESULT)
    documents_fetched = fetch_documents(document_indexes_needed)
    if documents_fetched:
        additional_docs_needed = number_per_page - len(documents_fetched)
    else:
        additional_docs_needed = number_per_page
    # If more documents are needed, fetch additional documents
    while additional_docs_needed > 0 and end_index < current_length:
        pagination_offset += additional_docs_needed
        # Adjust start and end indexes to fetch more documents
        new_start_index = end_index
        end_index = new_start_index + additional_docs_needed

        additional_indexes_needed = CURRENT_RESULT[new_start_index:end_index]

        additional_documents = fetch_documents(additional_indexes_needed)
        if additional_documents:
            documents_fetched.extend(additional_documents)
        additional_docs_needed = number_per_page - len(documents_fetched)

        # Update buffer size or break condition to avoid infinite loops if needed
    if not documents_fetched:
        documents_fetched = {}
    return documents_fetched


def run_periodically():
    while True:
        schedule.run_pending()
        time.sleep(1)
        # Wait for 24 hours (86400 seconds)


load_dotenv()

nltk.download('stopwords')

PG_CONNECTION_CONFIG = {
    'host': os.getenv('PG_HOSTNAME'),
    'dbname': os.getenv('PG_DB_NAME'),
    'user': os.getenv('PG_USERNAME'),
    'port': os.getenv('PG_POST'),
    'password': os.getenv('PG_PASSWORD'),
}
REDIS_CONNECTION_CONFIG = {
    'host': os.getenv('REDIS_HOST'),
    'port': os.getenv('REDIS_PORT'),
    'password': os.getenv('REDIS_PASSWORD'),
    'decode_responses': True,
}

r = redis.Redis(**REDIS_CONNECTION_CONFIG)

connection = psycopg2.connect(**PG_CONNECTION_CONFIG)
DOC_IDS = []
N = 0
ID2DATE = {}

parsed_cache = {}

min_date_value = parse_date("0001-01-01", datetime.now())
parsed_cache = {"": min_date_value}

CURRENT_RESULT = []
compiled_patterns = {}
stemmers = {}
stemmed_languages = ["arabic", "danish", "dutch", "english", "finnish", "french", "german", "hungarian", "italian",
                     "norwegian", "portuguese", "romanian", "russian", "spanish", "swedish"]
non_alpha_pattern = regex.compile(r'\p{P}')

non_alpha_pattern_boolean = regex.compile(r'[^\w\s#"()]+')
logical_operators = {'AND', 'OR', 'NOT'}

tokenization_regex_boolean = r'(\bAND\b|\bOR\b|\bNOT\b|\b\w+\b|[#"\(\)])'

pagination_offset = 0

schedule.every().day.at("06:00").do(update_database_info)

update_database_info()
thread = threading.Thread(target=run_periodically)
# Daemon threads are killed when the main program exits
thread.daemon = True
thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="info")
