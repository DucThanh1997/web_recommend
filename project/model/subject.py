from bson.json_util import dumps
import json
from common import constant
from model import DB



class Subject(object):
    @staticmethod
    def insert(data):
        DB.insert(collection=constant.MONGO_COLLECTION_SUBJECT, data=data)

    @staticmethod
    def find_one(query):
        return DB.find_one(collection=constant.MONGO_COLLECTION_SUBJECT, query=query)

    @staticmethod
    def delete(id):
        bson.objectid.ObjectId.is_valid(id)
        query = {"_id": id}
        DB.delete_one(collection=constant.MONGO_COLLECTION_SUBJECT, query=query)

    @staticmethod
    def find(query):
        document = DB.find_all(
            collection=constant.MONGO_COLLECTION_SUBJECT, query=query
        )
        subjects = dumps(document)
        
        subjects = subjects.replace("'", "\"")
        subjects = json.loads(subjects)
        return subjects

    @staticmethod
    def update_one(query, change):
        DB.update_one(
            collection=constant.MONGO_COLLECTION_SUBJECT,
            query=query,
            newvalue=change
        )
