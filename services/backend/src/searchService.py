import re

from numpy import log10
from logging import getLogger
from fastapi import Request

from .utils.preprocessing import preprocess_query, BOOL_OPERATORS
from .utils.constants import RESULTS_PAGE_SIZE
from .redisService import get_index
from .booleanSearch import boolean_search

logger = getLogger('uvicorn')


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


def ranked_search(tokens: list[str], n_docs):
    scores = {}
    for term, document_entries in get_index(tokens).items():
        if not document_entries:
            continue
        idf = log10(n_docs / len(document_entries.keys()))
        for doc_id, idx_term in document_entries.items():
            # date_factor = ID2DATE.get(doc_id, ID2DATE[''])
            doc_id = int(doc_id)
            tf = 1 + log10(len(idx_term.split(",")))
            scores[doc_id] = scores.get(doc_id, 0) + (tf * idf)
    return sorted(scores, key=scores.get, reverse=True)


async def search(query, request: Request, page: int = 1, size: int = RESULTS_PAGE_SIZE):
    _query = preprocess_query(query.split(' '))
    _offset = (page * size) - size
    # BOOLEAN SEARCH
    if re.search('|'.join(BOOL_OPERATORS), ' '.join(_query)):
        return boolean_search(query.split(' '), DOC_IDS)
    # RANKED SEARCH
    else:
        n_docs = await request.app.state.db.fetch_rows('SELECT count(*) as count FROM jobs')
        doc_ids = ranked_search(_query, n_docs[0].get('count'))
        results = await request.app.state.db.fetch_rows(
            f'SELECT * FROM jobs WHERE id in ({",".join([str(d) for d in doc_ids[_offset:_offset + size]])})'
        )
        return results
