from __future__ import division
from flask_restful import reqparse, Resource
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from flask_restful import reqparse, Resource
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.datastructures import FileStorage
from sklearn import tree

from utils.train_sample_processing import Processing_data_knn, Processing_data_naive
from utils.train_processing import save_training_to_mongo, saved_score, save_model, save_header


class TrainingSample(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        parser.add_argument('thuat_toan',
                         type=str, 
                         required=True,
                         help="ko duoc bo trong")
        parser.add_argument('training_percent',
                         type=int, 
                         required=True,
                         help="ko duoc bo trong")
        parser.add_argument('testing_percent',
                         type=int, 
                         required=True,
                         help="ko duoc bo trong")
        parser.add_argument('phan_phoi',
                         type=str, 
                         help="ko duoc bo trong")
        parser.add_argument('neighbour',
                         type=str, 
                         help="ko duoc bo trong")
        data = parser.parse_args()
        print("data: ", data)
        try:
            dataToPredict = pd.read_csv(data["resource_csv"])
        except: 
            return {
                "msg": "Bad request"
            }, 400
        testing = float(data["testing_percent"]) / 100
        header = list(dataToPredict.columns.values)
        if data["thuat_toan"] == "Knn":
            if data["neighbour"] == '':
                neighbour = 3
            else:
                neighbour = int(data["neighbour"])

            testing = float(data["testing_percent"]) / 100
            score = TrainKnnSample(data= dataToPredict, 
                                   neighbour= neighbour, 
                                   training= data["training_percent"] / 100,
                                   testing= testing)

            

        elif data["thuat_toan"] == "naive":
            print("phan_phoi: ", data["phan_phoi"])
            print("testing_percent: ", data["testing_percent"])
            print("testing_percent: ", type(data["testing_percent"]))
            testtt = float(20 / 100)
            print("test: ", testtt)
            score = trainNaiveBayesSample(data=dataToPredict, 
                                    training=data["training_percent"],
                                    testing=testtt)

        # else:
        #     print("1: ", data["phan_phoi"])
        #     data["training_percent"]
        #     score = TrainID3(data=dataToPredict, 
        #                      name=data["khoa"], 
        #                      training=data["training_percent"],
        #                      testing=data["testing_percent"])

        result_save_header = save_header(headers=header[:-1], khoa="sample", thuat_toan=data["thuat_toan"])
        if result_save_header != "okke":
            return {
                "msg": "Bad request"
            }, 400
            
        if score == 0:
            return {
                "msg": "Bad request"
            }, 400

        return {
            "score": score
        }, 200


def TrainKnnSample(data, neighbour, training, testing):
    
    a = data.shape

    data_changed = Processing_data_knn(data=data, columns=a[1])
    print("change data okke")
    if type(data_changed) is int:
        print("van de data")
        return 0

    label = data_changed.iloc[:-1, -1].values
    features = data_changed.iloc[:-1, :-1].values
    
    print("neighbour: ", neighbour)

    classifier = KNeighborsClassifier(n_neighbors=int(neighbour))
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=testing)

    result_save_to_db = save_training_to_mongo(train=X_train, 
                                               label=y_train,
                                               thuat_toan="knn",
                                               khoa="sample")
    if result_save_to_db != "okke":
        return 0

    classifier.fit(X_train, y_train.astype('int'))
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test.astype('int'), y_pred)
    round_score = saved_score(score=score, model_name="sample" + "_" + "knn")

    result_save_to_db = save_model(classifier=classifier,
                                   thuat_toan="knn",
                                   khoa="sample")
    if result_save_to_db != "okke":
        return 0
    return round_score

    

def trainNaiveBayesSample(data, training, testing):
    a = data.shape
    data_changed = Processing_data_naive(data=data, columns=a[1])
    print("change data okke")
    if type(data_changed) is int:
        print("van de data")
        return 0

    label = data_changed.iloc[:-1, -1].values
    features = data_changed.iloc[:-1, :-1].values

    classifier = GaussianNB()
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=testing)

    result_save_to_db = save_training_to_mongo(train=X_train, 
                                               label=y_train,
                                               thuat_toan="naive",
                                               khoa="sample")
    if result_save_to_db != "okke":
        return 0

    classifier.fit(X_train, y_train.astype('int'))
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test.astype('int'), y_pred)

    round_score = saved_score(score=score, model_name="sample" + "_" + "naive")

    result_save_to_db = save_model(classifier=classifier,
                                   thuat_toan="naive",
                                   khoa="sample")
    if result_save_to_db != "okke":
        return 0
    return round_score