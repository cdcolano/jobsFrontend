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

JOBS_POOL_CURSOR_NAME = os.getenv('JOBS_POOL_CURSOR_NAME')
JOBS_POOL_CURSOR_SIZE = os.getenv('JOBS_POOL_CURSOR_SIZE')


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
    return index


def update_remote_index(index: dict) -> bool:
    with redis.Redis(**REDIS_CONNECTION_CONFIG) as rd_connection:
        for term, data in index.items():
            rd_connection.hset(term, mapping=data)
    return True


if __name__ == '__main__':
    start_time = time.time()

    with psycopg2.connect(**PG_CONNECTION_CONFIG) as pg_connection:
        with pg_connection.cursor() as cursor:
            cursor.execute("SELECT Count(id) FROM jobs;")
            N = cursor.fetchone()[0]
            print(N)

        with pg_connection.cursor() as jp_cursor:
            jp_cursor.execute("DECLARE JOBS_POOL_CUR CURSOR FOR SELECT * FROM fetch_jobs_content;")
            n = 0
            while n < N:
                jp_cursor.execute(f"FETCH FORWARD {JOBS_POOL_CURSOR_SIZE} FROM JOBS_POOL_CUR;")
                update_remote_index(build_index(jp_cursor.fetchall()))
                n += JOBS_POOL_CURSOR_SIZE
                print(f"{(n / N) * 100}%")

    print("--- %s seconds ---" % (time.time() - start_time))
