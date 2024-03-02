import os
import time
import psycopg2
import redis

from preprocessing import preprocess
from dotenv import load_dotenv

load_dotenv()

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

FETCH_ALL_JOBS_VIEW = os.getenv('FETCH_ALL_JOBS_VIEW')

JOBS_POOL_CURSOR_NAME = os.getenv('JOBS_POOL_CURSOR_NAME', default='JOBS_POOL_CURSOR')
JOBS_POOL_CURSOR_SIZE = int(os.getenv('JOBS_POOL_CURSOR_SIZE', default=2000))

JOB_INDEX_START_POSITION = int(os.getenv('JOBS_INDEX_START_POSITION', default=0))
JOBS_INDEX_END_POSITION = os.getenv('JOBS_INDEX_END_POSITION', default=None)


def build_index(documents: list[list[str]]) -> dict:
    """
    Builds the index for the given document

    Parameters
    ----------
    documents : list[str]
        contains the content of the documents.

    Returns
    -------
    index : dict
        dictionary containing the built index.
    """
    index = {}
    for doc in documents:
        for pos, term in enumerate(preprocess((doc[1]))):
            if term not in index.keys():
                index[term] = {doc[0]: str(pos)}
            elif doc[0] not in index[term].keys():
                index[term][doc[0]] = str(pos)
            else:
                index[term][doc[0]] += f',{pos}'
    del documents
    return index


def update_remote_index(index: dict) -> bool:
    with redis.Redis(**REDIS_CONNECTION_CONFIG) as rd_connection:
        pipe = rd_connection.pipeline()
        [pipe.hset(term, mapping=data) for term, data in index.items()]
        pipe.execute()
    del index
    return True


def index_full_database(offset: int = 0) -> bool:
    with psycopg2.connect(**PG_CONNECTION_CONFIG) as pg_connection:
        with pg_connection.cursor() as jp_cursor:
            jp_cursor.execute(f"""
            SELECT * FROM {FETCH_ALL_JOBS_VIEW} 
            WHERE id > {offset} AND id <= {offset + JOBS_POOL_CURSOR_SIZE}  
            ORDER BY id;
            """)
            update_remote_index(build_index(jp_cursor.fetchall()))
    return True


if __name__ == '__main__':
    start_time = time.time()

    N = int(JOBS_INDEX_END_POSITION) if JOBS_INDEX_END_POSITION else None
    if not N:
        with psycopg2.connect(**PG_CONNECTION_CONFIG) as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT Count(id) FROM jobs;")
                N = int(cursor.fetchone()[0])
                print(N)

    n_processed_docs = JOB_INDEX_START_POSITION
    while n_processed_docs < N:
        t = time.time()
        try:
            index_full_database(n_processed_docs)
            n_processed_docs += JOBS_POOL_CURSOR_SIZE
        except Exception as e:
            print(e)
            print("PG DB Connection lost, reconnecting ...")
        print(f"{n_processed_docs:7}/{N:7} | "
              f"{(n_processed_docs / N) * 100:3.2f}% | "
              f"execution time:{time.time() - t:.2f} seconds"
              )

    print("--- %s seconds ---" % (time.time() - start_time))
