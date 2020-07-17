from bson.json_util import dumps
import json
from common import constant
from model import DB



class Score(object):
    @staticmethod
    def insert(data):
        DB.insert(collection=constant.MONGO_COLLECTION_SCORE, data=data)

    @staticmethod
    def find_one(query):
        score = DB.find_one(collection=constant.MONGO_COLLECTION_SCORE, query=query)
        
        if score is None:
            return -1
        print("score: ", score["score"])
        return score["score"]

    @staticmethod
    def find_one_sample(query):
        score = DB.find_one(collection=constant.MONGO_COLLECTION_SCORE, query=query)
        if score is None:
            return -1
        print("score: ", score["score"])
        return score["score"], score["max"], score["min"]

    @staticmethod
    def delete(id):
        # bson.objectid.ObjectId.is_valid(id)
        query = {"id": id}
        DB.delete_one(collection=constant.MONGO_COLLECTION_SCORE, query=query)

    @staticmethod
    def find(query):
        document = DB.find_all(
            collection=constant.MONGO_COLLECTION_SCORE, query=query
        )
        subjects = dumps(document)
        
        subjects = subjects.replace("'", "\"")
        subjects = json.loads(subjects)
        return subjects

    @staticmethod
    def update_one(query, change):
        return DB.update_one(
                collection=constant.MONGO_COLLECTION_SCORE,
                query=query,
                values=change
            )
