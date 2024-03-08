from numpy import log10
from utils.preprocessing import preprocess_query, BOOL_OPERATORS
from redisService import get_index


# def boolean_search(query: list[str], index: str, verbose: bool = False) -> pd.DataFrame:
#     """
#     Do boolean search to specified query.
#
#     Does proximity search, phrase search, and applies boolean operators.
#
#     Parameters
#     ----------
#     query : list[str]
#         query to search for.
#     index : str
#         Name of index file to load from the INDEX_PATH folder.
#     verbose : bool, optinal
#         Indicates to have verbose output (defaults to `False`).
#
#     Returns
#     -------
#     results : pd.DataFrame
#         pandas dataframe containing the results fr the search.
#     """
#
#     def proximity_search(query: list[str], index: pd.DataFrame, max_distance: int = PROXIMITY_MAX_DISTANCE,
#                          phrase_search: bool = False) -> pd.DataFrame:
#         """
#         Does proximity search for the specifies query in the specified index
#
#         Parameters
#         ----------
#         query : list[str]
#             Query to search for.
#         index : pd.DataFrame
#             Index to search into.
#         max_distance : int, optional
#             Indicates the maximum distance to consider during the search.
#         phrase_search : bool, optional
#             Triggers phrase search. Sets max_distance to 1. defaults to (`False`)
#
#         Returns
#         -------
#         results : pd.DataFrame
#             pandas dataframe containing the results fr the search.
#         """
#         _query = query
#         query = []
#         for q in _query:  # Get all the terms on query
#             [query.append(x) for x in q.split(' ')]
#
#         results = pd.DataFrame({  # Get all the hits for every term
#             "QUERY": query,
#             "DOCNOS": [','.join({str(x) for x in search_term(term, index)}) for term in query if
#                        term not in BOOL_OPERATORS]
#         })
#
#         if len(query) <= 1: return results  # If query contains just 1 term interrupt execution and return results
#
#         doc_list = set()
#         [doc_list.update({y for y in x.split(',')}) for x in results.DOCNOS if
#          x]  # Get the set of all the documents that hit the terms
#
#         bigrams = []
#         for i in range(1, len(query)):  # Build bigrams for proximity search
#             bigrams.append((query[i - 1], query[i]))
#
#         terms_positions = pd.DataFrame(
#             [  # Build DF containing document number, bigram, position of term 1 and position of term 2
#                 (
#                     doc_number,
#                     '_'.join(bigram),
#                     row[bigram[0]] if bigram[0] in index.columns.values else np.NaN,
#                     row[bigram[1]] if bigram[1] in index.columns.values else np.NaN
#                 )
#                 for bigram in bigrams
#                 for doc_number, row in index[index.index.isin(doc_list)].iterrows()
#             ],
#             columns=['DOCNO', 'TERMS', 'T1_POS', 'T2_POS']
#         ).dropna().drop_duplicates().reset_index(drop=True)  # drop empty rows and duplicates
#
#         terms_positions['DISTANCES'] = [
#             [  # calculate distances between terms
#                 int(t2) - int(t1) for t2 in row['T2_POS'].split(',')
#                 for t1 in row['T1_POS'].split(',')
#             ] for _, row in terms_positions.iterrows()
#         ]
#
#         terms_positions['RELEVANCE'] = terms_positions['DISTANCES'].apply(  # calculate relevancy by proximity criteria
#             lambda x: any([0 < abs(y) <= max_distance for y in x]) if not phrase_search else any(
#                 [0 < y <= 1 for y in x])
#         )
#
#         terms_positions = terms_positions[terms_positions['RELEVANCE']]  # Filter only the relevant results
#
#         if not terms_positions.empty:
#             positive_docs = set()  # build the set of hits
#             bigram_doc_list = set(terms_positions['DOCNO'].values)
#             for bigram_doc in bigram_doc_list:
#                 if len(terms_positions[terms_positions['DOCNO'] == bigram_doc].index) >= len(
#                         query) - 1: positive_docs.add(bigram_doc)
#
#         return pd.DataFrame({  # build the response dataframe
#             "QUERY": [' '.join(query)],
#             "DOCNOS": [','.join({str(x) for x in positive_docs}) if not terms_positions.empty else None]
#         })
#
#     def apply_NOT(results: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply the NOT boolean operator to the guiven query
#
#         Parameters
#         ----------
#         results : pd.DataFrame
#             pandas dataframe containing the results of the search.
#         index : pd.DataFrame
#             pandas dataframe containing the loaded index.
#
#         Returns
#         -------
#         results : pd.DataFrame
#             pandas dataframe containing the results of the search after
#             applying the NOT operator.
#         """
#         for i, row in results.iterrows():
#             if row.QUERY == NOT:
#                 prev_row = results.iloc[i - 1]
#                 next_row = results.iloc[i + 1]
#                 next_row.DOCNOS = ','.join([str(x) for x in search_term(next_row.QUERY, index, compliment=True)])
#                 next_row.QUERY = ' '.join([NOT, next_row.QUERY])
#                 if i > 0 and prev_row.QUERY not in BOOL_OPERATORS:
#                     row.QUERY = AND
#         return results[results.QUERY != NOT].reset_index(drop=True)
#
#     def apply_AND_and_OR(results: pd.DataFrame) -> pd.DataFrame:
#         """
#         Apply the AND and OR boolean operators to the guiven query
#
#         Parameters
#         ----------
#         results : pd.DataFrame
#             pandas dataframe containing the results of the search.
#
#         Returns
#         -------
#         results : pd.DataFrame
#             pandas dataframe containing the results of the search after
#             applying the AND and OR operators.
#         """
#         for i, row in results.iterrows():
#             if row.QUERY in BOOL_OPERATORS:
#                 if i == 0 or i == len(results) - 1:
#                     raise IOError(INVALID_QUERY_ERROR_MSGS['EDGE_OPERATORS'])
#                 _prev_row = results.iloc[i - 1]
#                 _next_row = results.iloc[i + 1]
#                 _prev_row_docnos = set(_prev_row.DOCNOS.split(','))
#                 _next_row_docnos = set(_next_row.DOCNOS.split(','))
#                 _prev_row_docnos.discard(''), _next_row_docnos.discard('')
#                 if row.QUERY == AND:
#                     row.DOCNOS = ','.join(
#                         [str(x) for x in _prev_row_docnos & _next_row_docnos])  # Apply the AND opertor
#                 elif row.QUERY == OR:
#                     row.DOCNOS = ','.join(
#                         [str(x) for x in _prev_row_docnos | _next_row_docnos])  # Apply the OR operator
#                 row.QUERY = ' '.join([_prev_row.QUERY, row.QUERY, _next_row.QUERY])
#                 results = results.drop([i - 1, i + 1]).reset_index(drop=True)
#                 break
#         return results
#
#     index, _ = load_index(index, preprocess_query(query), verbose=verbose)  # Load index
#
#     # Separate query in subqueries (or terms) and boolean operators
#     sub_queries = [x.strip() for x in
#                    ' '.join(query).replace(' AND ', ' % ').replace(' OR ', ' % ').replace(' NOT ', ' % ').split('%')]
#     bool_operators = list(filter(lambda x: x in BOOL_OPERATORS, query))
#
#     results = pd.DataFrame()
#     for sub_query in sub_queries:  # Do search for every subquery
#         if sub_query == '':  # If subquery is empty return empty dataframe
#             results = pd.concat([results, pd.DataFrame({'QUERY': [''], 'DOCNOS': ['']})]).reset_index(drop=True)
#             continue
#
#         # PROXIMITY SEARCH
#         prox_sub_qs = re.findall(PROX_REGEX, sub_query)
#         for prox_sub_q in prox_sub_qs:
#             p, q = prox_sub_q[:-1].split('(')
#             p = p.replace('#', '')
#             q = q.replace(' ', '').split(',')
#             results = pd.concat([results, proximity_search(preprocess_query(q), index, int(p))]).reset_index(drop=True)
#
#         # PHRASE SEARCH
#         phrase_sub_qs = re.findall(PHRASE_REGEX, sub_query)
#         for phrase_sub_q in phrase_sub_qs:
#             q = phrase_sub_q[1:-1]
#             q = q.split(' ')
#             results = pd.concat(
#                 [results, proximity_search(preprocess_query(q), index, phrase_search=True)]).reset_index(drop=True)
#
#         # Do proximity to remaining terms that don't match either proximity or phraase regex
#         if len(prox_sub_qs) == 0 and len(phrase_sub_qs) == 0:
#             results = pd.concat([results, proximity_search(preprocess_query(sub_query.split(' ')), index,
#                                                            phrase_search=True)]).reset_index(drop=True)
#
#     # Join again subqueries with boolean keywords
#     for i, bool_operator in enumerate(bool_operators):
#         results.loc[i + 0.5] = [bool_operator.replace(' ', ''), np.NaN]
#     results = results[results['QUERY'] != '']  # remove empty rows
#     results = results.sort_index().reset_index(drop=True)
#
#     results = apply_NOT(results, index)  # Apply NOT boolean operator
#
#     for _ in [x for x in query if x in BOOL_OPERATORS]:  # Apply AND and OR boolean operators
#         results = apply_AND_and_OR(results)
#
#     return results


def ranked_search(tokens: list[str], n_docs):
    scores = {}
    for document_entries in get_index(tokens):
        if not document_entries:
            continue
        idf = log10(n_docs / len(document_entries))
        for doc_id, idx_term in document_entries.items():
            # date_factor = ID2DATE.get(doc_id, ID2DATE[''])
            doc_id = int(doc_id)
            tf = 1 + log10(len(idx_term.split(",")))
            scores[doc_id] = scores.get(doc_id, 0) + (tf * idf)
    return sorted(scores, key=scores.get, reverse=True)


def search(query):
    _query = preprocess_query(query.split(' '))

    # BOOLEAN SEARCH
    # if re.search('|'.join(BOOL_OPERATORS), ' '.join(_query)):
    #     results = boolean_search(query)
    #     if results.empty: return None
    #     results = results.DOCNOS[0].split(',') if results.DOCNOS[0] else ''
    #     return results if '' not in results else results.remove('')
    # RANKED SEARCH
    # else:
    index = get_index(_query)
    # results = ranked_search(_query, 1000000)
    # return [str(x) + ',' + str(y) for x, y in results.items()]
    return index
