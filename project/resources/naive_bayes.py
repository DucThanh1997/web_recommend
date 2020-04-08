# coding=utf-8

from flask_restful import reqparse, Resource
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from werkzeug.datastructures import FileStorage
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from common import constant
import collections 


class RecommendSystem(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        data = parser.parse_args()

        try:
             dataToPredict = pd.read_csv(data["resource_csv"])
        except: 
            return {
                "msg": "Bad request"
            }, 400
        # du doan xep loai tot nghiep
        













class NaiveBayes(Resource):
    def post(self):
        le = LabelEncoder()
        parser = reqparse.RequestParser()
        parser.add_argument('yellow',
                    type=int,
                    required=True,
                    help='CSV file')
        parser.add_argument('long',
                    type=int,
                    required=True,
                    help='CSV file')
        parser.add_argument('sweet',
                         type=int,
                         required=True,
                         help='CSV file')
        data = parser.parse_args()

        # loc du lieu
        if (data["yellow"] != 0 and data["yellow"] != 1) or (data["long"] != 0 and data["long"] != 1) or (data["sweet"] != 0 and data["sweet"] != 1):
            return {
                "msg": "Bad request1"
            }, 400            
        # encode
        filename = "naive.pkl"
        try: 
            model = pickle.load(open(filename, 'rb'))
            predicted = model.predict([[data["long"],data["sweet"],data["yellow"]]]) 
        except Exception as e:
            return {
                "msg": "bad request"
            }, 400

        
        if str(predicted) == "[1]":
            print("str(predicted): ", str(predicted))
            return {
                "msg": "orange"
            }, 200
        elif str(predicted) == "[0]":
            print("str(predicted): ", str(predicted))
            return {
                "msg": "banana"
            }, 200
        else:
            print("str(predicted): ", str(predicted))
            return {
                "msg": "other"
            }



class TrainNaiveTest(Resource):
    def post(self):
        le = LabelEncoder()
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        data = parser.parse_args()

        try:
             dataToTrain = pd.read_csv(data["resource_csv"])
        except: 
            return {
                "msg": "Bad request"
            }, 400

        # print("1")
        # loc du lieu
        X = dataToTrain.iloc[:, 0:-1]
        # print("2")
        y = dataToTrain.iloc[:, -1]
        print("X: ", X)

        X1 = dataToTrain.iloc[1, 0:-1]
        print("X1: ", X1.shape)
        print("X1: ", X1)
        # train va du doan
        model = GaussianNB()
        model.fit(X, y)
        # print("3")
        pkl_filename = "naive.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        try:
            print("1: ", len(X_train))
            y_pred = model.predict(X_test)
            # print("y_test: ", y_test)
            # print("y_pred: ", y_pred)
            score = accuracy_score(y_test, y_pred)
        except Exception as e:
            print("err: ", e)
            return {
                "msg": "bad request",
            }, 400
        print("score: ", score)
        return {
            "msg": "okke",
            "score": score
        }, 200


class TrainNaive(Resource):
    def post(self):
        le = LabelEncoder()
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        data = parser.parse_args()

        try:
             dataToTrain = pd.read_csv(data["resource_csv"])
        except: 
            return {
                "msg": "Bad request"
            }, 400
        # loc du lieu
        le = LabelEncoder()

        # lay ra ten nhan va bien no thanh 0,1,2
        value = dataToTrain.iloc[:, 0].values
        changed = le.fit_transform(value)
        dataToTrain.iloc[:, 0] = changed

        # set nhan va tinh nang
        label = dataToTrain.iloc[:, 0]
        features = dataToTrain.iloc[:, 1:]


        # train va du doan
        model = GaussianNB()
        model.fit(features, label)

        pkl_filename = "naive.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

        X_train, X_test, y_train, y_test = train_test_split(features, label)
        try:
            print("1: ", len(X_test))
            y_pred = model.predict(X_test)
            print("y_test: ", y_test)
            print("y_pred: ", y_pred)
            score = accuracy_score(y_test, y_pred)
        except Exception as e:
            print("err: ", e)
            return {
                "msg": "bad request",
            }, 400
        print("score: ", score)
        return {
            "msg": "okke",
            "score": score
        }, 200