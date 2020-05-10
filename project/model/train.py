from bson.json_util import dumps
import json
from common import constant
from model import DB



class Train(object):
    @staticmethod
    def insert(data):
        DB.insert(collection=constant.MONGO_COLLECTION_TRAIN, data=data)

    @staticmethod
    def find_one(query):
        train = DB.find_one(collection=constant.MONGO_COLLECTION_TRAIN, query=query)
        if train is None:
            return -1
        return train["label"]

    @staticmethod
    def update_one(query, change):
        return DB.update_one(
                collection=constant.MONGO_COLLECTION_TRAIN,
                query=query,
                values=change
            )
