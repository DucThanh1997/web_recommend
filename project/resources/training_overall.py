#!/usr/bin/env python
# -*- coding: utf-8 -*- 


from utils.train_processing import save_training_to_mongo, saved_score, save_model, save_header

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
from sklearn.model_selection import cross_val_score
from statistics import mean


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
                         required=True, 
                         help="ko duoc bo trong")
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
        parser.add_argument('neighbour',
                         type=str, 
                         help="ko duoc bo trong")
        data = parser.parse_args()
        print("data: ", data)
        
        try:
            dataToPredict = pd.read_csv(data["resource_csv"])
            header = list(dataToPredict.columns.values)
        except: 
            return {
                "msg": "Bad request"
            }, 400
        if data["thuat_toan"] == "knn":
            if data["neighbour"] == '':
                neighbour = 3
            else:
                neighbour = int(data["neighbour"])

            testing = float(data["testing_percent"]) / 100
            score = TrainKnn(data= dataToPredict, 
                             name= data["khoa"], 
                             neighbour= neighbour, 
                             training= data["training_percent"])
            header = list(dataToPredict.columns.values)

        elif data["thuat_toan"] == "naive":
            testing = float(data["testing_percent"]) / 100
            score = TrainNaiveBayes(data=dataToPredict, 
                                    name=data["khoa"], 
                                    training=data["training_percent"],
                                    testing=testing)

        else:
            data["training_percent"]
            score = TrainID3(data=dataToPredict, 
                             name=data["khoa"], 
                             training=data["training_percent"])
        result_save_header = save_header(headers=header[1:-1], khoa=data["khoa"], thuat_toan=data["thuat_toan"])
        if result_save_header != "okke":
            return {
                "msg": "Bad request"
            }, 400
            
        if score == -1:
            return {
                "msg": "Bad request"
            }, 400

        return {
            "score": score
        }, 200

def TrainKnn(data, name, neighbour, training):
    features = data.iloc[:-1, 1:-1].values
    label = data.iloc[:-1, -1].values
    classifier = KNeighborsClassifier(n_neighbors=neighbour, weights="distance", p=2)

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.05)
    # save traine_model
    result_save_to_db = save_training_to_mongo(train=X_train, 
                                               label=y_train,
                                               thuat_toan="knn",
                                               khoa=name)
    # print("result_save_to_db: ", result_save_to_db)
    # if result_save_to_db != "okke":
    #     return -1


    scores = cross_val_score(classifier, features, label, cv=training)
    classifier.fit(X_train, y_train.astype('int'))
    y_pred = classifier.predict(X_test)
    score = accuracy_score(y_test.astype('int'), y_pred)
    round_score = saved_score(score=max(scores), model_name=name + "_" + "knn")
    print("scores: ", scores)
    print("mean: ", mean(scores))
    print("round score: ", round_score)
    print("score: ", score)

    result_save_to_db = save_model(classifier=classifier,
                                   thuat_toan="knn",
                                   khoa=name)
    if result_save_to_db != "okke":
        return 0
    return round_score


def TrainNaiveBayes(data, name, training, testing):
    # loc du lieu
    X = data.iloc[:, 1:-1].values
    y = data.iloc[:, -1].values

    # train va du doan
    try:
        print("1")
        model = GaussianNB()
        model.fit(X, y)
        print("2")
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing)
        # result_save_to_db = save_training_to_mongo(train=X_train, 
        #                                     label=y_train,
        #                                     thuat_toan="naive",
        #                                     khoa=name)
        # if result_save_to_db != "okke":
        #     return 0

        scores = cross_val_score(model, X, y, cv=training)
        print("scores: ", scores)
        round_score = saved_score(score=max(scores), model_name=name + "_" + "naive")

        result_save_to_db = save_model(classifier=model,
                                       thuat_toan="naive",
                                       khoa=name)
        if result_save_to_db != "okke":
            return 0
        print("okke result_save_to_db")
    except Exception as e:
        print("err: ", e)
        score = 0
        return score
    return round_score

def TrainID3(data, name, training):
    features = data.iloc[:-1, 1:-1].values
    label = data.iloc[:-1, -1].values

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.05)
    decisionTree = tree.DecisionTreeClassifier(max_depth=None, criterion='entropy', class_weight=None)
    decisionTree.fit(X_train, y_train)
    # result_save_to_db = save_training_to_mongo(train=X_train, 
    #                                     label=y_train,
    #                                     thuat_toan="naive",
    #                                     khoa=name)
    # if result_save_to_db != "okke":
    #     return 0

    scores = cross_val_score(decisionTree, features, label, cv=training)

    # y_pred = decisionTree.predict(X_test)
    # score = accuracy_score(y_test, y_pred)


    result_save_to_db = save_model(classifier=decisionTree,
                                   thuat_toan="id3",
                                   khoa=name)
    if result_save_to_db != "okke":
        return 0

    # lưu score
    round_score = saved_score(score=max(scores), model_name=name + "_" + "id3")
    print("scores: ", scores)
    print("mean: ", mean(scores))
    print("round score: ", round_score)
    # print("score: ", score)
    return round_score

