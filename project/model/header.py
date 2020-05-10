from bson.json_util import dumps
import json
from common import constant
from model import DB



class Header(object):
    @staticmethod
    def insert(data):
        DB.insert(collection=constant.MONGO_COLLECTION_HEADER, data=data)

    @staticmethod
    def find_one(query):
        headers = DB.find_one(collection=constant.MONGO_COLLECTION_HEADER, query=query)
        if headers is None:
            return -1
        return headers["header"]

    @staticmethod
    def update_one(query, change):
        return DB.update_one(
                collection=constant.MONGO_COLLECTION_HEADER,
                query=query,
                values=change
            )