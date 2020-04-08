from flask_restful import reqparse, Resource
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from common import constant
import collections 
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.datastructures import FileStorage
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from sklearn import utils


class TrainingOverall(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        parser.add_argument('khoa',
                         type=str, 
                         help="ko duoc bo trong")
        parser.add_argument('thuat_toan',
                         type=str, 
                         help="ko duoc bo trong")
        data = parser.parse_args()

        if data["khoa"] == "" or data["thuat_toan"] == "":
            return {
                "msg": "Bad request"
            }, 400
        try:
             dataToPredict = pd.read_csv(data["resource_csv"])
        except: 
            return {
                "msg": "Bad request"
            }, 400
        if data["thuat_toan"] == "Knn":
            score = TrainKnn(data=dataToPredict, name=data["khoa"], neighbour=3)
            if score == 0:
                return {
                    "msg": "Bad request"
                }, 400

            return {
                "score": score
            }, 200



def TrainKnn(data, name, neighbour):
    features = data.iloc[:-1, 1:-1].values
    # print("features: ", features)
    label = data.iloc[:-1, -1].values


    classifier = KNeighborsClassifier(n_neighbors=neighbour)
    # print("label: ", label)
    # print(len(label))
    
    X_train, X_test, y_train, y_test = train_test_split(features, label)
    # print("y: ", y_test)
    # print(X_train)
    classifier.fit(features, label.astype('int'))
    print("4")
    try:
        # print("1: ", len(X_test))
        y_pred = classifier.predict(X_test)
        pkl_filename = name + "_" + "knn" + ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(classifier, file)
        print("2")
        score = accuracy_score(y_test.astype('int'), y_pred)
    except Exception as e:
        print("err: ", e)
        return 0
    return score