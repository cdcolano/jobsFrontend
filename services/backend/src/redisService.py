import redis

from config.redis_config import pool


def get_index(terms):
    with redis.Redis(connection_pool=pool) as redis_client:
        pipe = redis_client.pipeline()
        [pipe.hgetall(term) for term in terms]
        return pipe.execute()
