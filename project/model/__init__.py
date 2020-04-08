import pymongo
from common import utils, constant
import datetime


def serializable(o):
    for k, v in o.items():
        if type(v) is datetime.datetime:
            o[k] = v.strftime(constant.YY_MM_DD_HH_MM_SS)
    del o["_id"]
    return o


class DB:
    @staticmethod
    def init(db_host, db_user, db_password, db_authen, db_name):
        client = pymongo.MongoClient(
            db_host, username=db_user, password=db_password, authSource=db_authen
        )
        client.server_info()
        DB.DATABASE = client[db_name]

    @staticmethod
    def insert(collection, data):
        DB.DATABASE[collection].insert(data)

    @staticmethod
    def find_one(collection, query):
        return DB.DATABASE[collection].find_one(query)
    
    @staticmethod
    def upsert(collection, query, values):
        return DB.DATABASE[collection].update_one(query, values, upsert=True)

    @staticmethod
    def update_one(collection, query, values):
        return DB.DATABASE[collection].update_one(query, values)

    @staticmethod
    def find_all(collection, query, limit=10000, skip=0, sort=1, sort_by=None):
        if sort_by is not None:
            return (
                DB.DATABASE[collection]
                .find(query)
                .limit(limit)
                .skip(skip)
                .sort(sort_by, sort)
            )
        return DB.DATABASE[collection].find(query).limit(limit).skip(skip)

    @staticmethod
    def count(collection, query):
        return DB.DATABASE[collection].count_documents(query)
