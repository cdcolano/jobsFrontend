import time
import nltk
import uvicorn
import schedule
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from dotenv import load_dotenv
from threading import Thread

from .config.pg_config import Database
from .utils.thesaurus import job_posting_thesaurus
from .utils.dateparser import parse_date
from .searchService import search
from .postgresService import get_database_constants

load_dotenv()

nltk.download('stopwords')

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


async def update_database_info(db: Database = None):
    try:
        if db is not None:
            app.state.N, app.state.DOC_IDS, app.state.ID2DATE = await get_database_constants(db, app.state.parsed_cache)
        else:
            app.state.N, app.state.DOC_IDS, app.state.ID2DATE = await get_database_constants(app.state.db, app.state.parsed_cache)
        logger.info(f'Updated database info.')
    except Exception as e:
        logger.exception(e)


@app.on_event("startup")
async def startup_event():
    app.state.parsed_cache = {}
    app.state.parsed_cache = {"": parse_date("0001-01-01", datetime.now(), app.state.parsed_cache)}
    logger.info('Created app state parsed-dates cache.')

    database_instance = Database()
    await database_instance.connect()
    await update_database_info(database_instance)
    app.state.db = database_instance
    logger.info('Created database connection pool.')

    logger.info("Server Startup")


@app.on_event("shutdown")
async def shutdown_event():
    if not app.state.db:
        await app.state.db.close()
    logger.info("Server Shutdown")


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


def expand_query(query):
    new_query = query
    for word in query.split():
        if word in job_posting_thesaurus:
            for synonym in job_posting_thesaurus[word]:
                new_query = new_query + " " + synonym
    return new_query


@app.get("/search/")
async def do_search(query: str, request: Request, page: int = 1):
    _start_time = time.time()
    if page < 1:
        raise HTTPException(status_code=400, detail='Page value must be equal or greater than 1.')
    try:
        results = await search(query, request, page=page)
        return {
            'processing': time.time() - _start_time,
            'data': [
                query,
                [{k: v for k, v in x.items()} for x in results]
            ]
        }
    except Exception as e:  # General / Unknown error
        logger.exception(e)
        raise HTTPException(status_code=500, detail={'query': query, 'message': f'Internal server error.'})


# def fetch_documents(indexes):
#     indexes_str = ','.join([f"'{index}'" for index in indexes])
#     query = f"""
#         SELECT json_agg(j ORDER BY j.idx)
#         FROM (
#             SELECT j.*, x.idx
#             FROM unnest(ARRAY[{indexes_str}]::int[]) WITH ORDINALITY AS x(id, idx)
#             JOIN jobs j ON j.id = x.id
#         ) AS j;
#
#     """
#
#     # Execute the query
#     cursor = connection.cursor()
#     cursor.execute(query)
#     result = cursor.fetchone()[0]  # Fetch the JSON result
#
#     cursor.close()
#     return result


# @app.get("/jobs/")
# async def retrieve_jobs(page: int, number_per_page: int):
#     global pagination_offset
#     start_index = (page - 1) * number_per_page
#     end_index = start_index + number_per_page
#
#     document_indexes_needed = CURRENT_RESULT[start_index:end_index]
#     current_length = len(CURRENT_RESULT)
#     documents_fetched = fetch_documents(document_indexes_needed)
#     if documents_fetched:
#         additional_docs_needed = number_per_page - len(documents_fetched)
#     else:
#         additional_docs_needed = number_per_page
#     # If more documents are needed, fetch additional documents
#     while additional_docs_needed > 0 and end_index < current_length:
#         pagination_offset += additional_docs_needed
#         # Adjust start and end indexes to fetch more documents
#         new_start_index = end_index
#         end_index = new_start_index + additional_docs_needed
#
#         additional_indexes_needed = CURRENT_RESULT[new_start_index:end_index]
#
#         additional_documents = fetch_documents(additional_indexes_needed)
#         if additional_documents:
#             documents_fetched.extend(additional_documents)
#         additional_docs_needed = number_per_page - len(documents_fetched)
#
#         # Update buffer size or break condition to avoid infinite loops if needed
#     if not documents_fetched:
#         documents_fetched = {}
#     return documents_fetched


# def run_periodically():
#     while True:
#         schedule.run_pending()
#         time.sleep(1)
#         # Wait for 24 hours (86400 seconds)


# DOC_IDS = []
# N = 0
# ID2DATE = {}

# CURRENT_RESULT = []
# stemmed_languages = ["arabic", "danish", "dutch", "english", "finnish", "french", "german", "hungarian", "italian",
#                      "norwegian", "portuguese", "romanian", "russian", "spanish", "swedish"]
# non_alpha_pattern = regex.compile(r'\p{P}')
#
# non_alpha_pattern_boolean = regex.compile(r'[^\w\s#"()]+')
# logical_operators = {'AND', 'OR', 'NOT'}

schedule.every().day.at("06:00").do(update_database_info)

# # update_database_info()
# thread = Thread(target=run_periodically)
# # Daemon threads are killed when the main program exits
# thread.daemon = True
# thread.start()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006, log_level="info")
