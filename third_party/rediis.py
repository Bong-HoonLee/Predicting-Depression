import redis

class Redis:
    _instance = None

    def __init__(self):
        self.client = redis.Redis(host='127.0.0.1', port=6379, db=0)

    @classmethod
    def get_instance(cls):
        if not cls._instance or not cls._instance.client or not cls._instance.client.ping():
            cls._instance = Redis()
        return cls._instance

    def ping(self):
        if self._instance.client is None:
            return False
        return self.get_instance().client.ping()

    def exists(self, key: str):
        if self._instance.client is None:
            return False
        return self.get_instance().client.exists(key)

    def expire(self, key, expire):
        if self._instance.client is None:
            return None
        return self.get_instance().client.expire(key, time=expire)

    def get(self, key):
        if self._instance.client is None:
            return None
        return self.get_instance().client.get(key)

    def set(self, key: str, value, expire=None):
        if self._instance.client is None:
            return None
        return self.get_instance().client.set(key, value, ex=expire)

    def hget(self, key: str, field: str, refresh: bool = True, expire: int = 0):
        if self._instance.client is None:
            return None

        result = self.get_instance().client.hget(name=key, key=field)
        if result is not None and refresh == True and expire > 0:
            self.expire(key, expire)

        return result

    def hmget(self, key: str, fields: list):
        if self._instance.client is None:
            return None
        return self.get_instance().client.hmget(key, fields)

    def hgetall(self, key: str):
        if self._instance.client is None:
            return None
        return self.get_instance().client.hgetall(key)

    def hset(self, key: str, mapping: dict, expire=None):
        if self._instance.client is None:
            return None
        result = self.get_instance().client.hset(name=key, mapping=mapping)
        if expire is not None:
            self.get_instance().client.expire(key, expire)
        return result

    def hmset(self, key: str, mapping: dict, expire=None):
        if self._instance.client is None:
            return None
        result = self.get_instance().client.hmset(name=key, mapping=mapping)
        if expire is not None:
            self.get_instance().client.expire(key, expire)
        return result

    def zadd(self, key: str, mapping: dict, expire=None):
        if self._instance.client is None:
            return None
        result = self.get_instance().client.zadd(name=key, mapping=mapping)
        if expire is not None:
            self.get_instance().client.expire(key, expire)
        return result

    def zincrby(self, key: str, member: str, score):
        if self._instance.client is None:
            return None
        result = self.get_instance().client.zincrby(name=key, value=member, amount=score)
        return result

    def zscore(self, key: str, member: str):
        if self._instance.client is None:
            return None
        return self.get_instance().client.zscore(name=key, value=member)
