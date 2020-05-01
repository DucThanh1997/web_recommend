from flask_restful import reqparse, Resource
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
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
from sklearn import tree


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
        if data["thuat_toan"] == "Knn":
            if data["neighbour"] == '':
                neighbour = 3
            else:
                neighbour = int(data["neighbour"])
            print("neighbour: ", neighbour)
            score = TrainKnn(data= dataToPredict, 
                             name= data["khoa"], 
                             neighbour= neighbour, 
                             training= data["training_percent"],
                             testing= data["testing_percent"])
            if score == 0:
                return {
                    "msg": "Bad request"
                }, 400

            return {
                "score": score
            }, 200
        elif data["thuat_toan"] == "NaiveBayes":
            print("phan_phoi: ", data["phan_phoi"])
            score = TrainNaiveBayes(data=dataToPredict, 
                                    name=data["khoa"], 
                                    training=data["training_percent"],
                                    testing=data["testing_percent"],
                                    phan_phoi=data["phan_phoi"])
            if score == 0:
                return {
                    "msg": "Bad request"
                }, 400

            return {
                "score": score
            }, 200

        elif data["thuat_toan"] == "ID3":
            print("1: ", data["phan_phoi"])
            data["training_percent"]
            score = TrainID3(data=dataToPredict, 
                             name=data["khoa"], 
                             training=data["training_percent"],
                             testing=data["testing_percent"])
            if score == 0:
                return {
                    "msg": "Bad request"
                }, 400

            return {
                "score": score
            }, 200



def TrainKnn(data, name, neighbour, training, testing):
    features = data.iloc[:-1, 1:-1].values
    # print("features: ", features)
    label = data.iloc[:-1, -1].values


    classifier = KNeighborsClassifier(n_neighbors=neighbour)
    # print("label: ", label)
    # print(len(label))
    
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=testing, train_size=training)
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


def TrainNaiveBayes(data, name, training, testing, phan_phoi):
    # loc du lieu
    X = data.iloc[:, 1:-1]
    # print("2")
    y = data.iloc[:, -1]

    X1 = data.iloc[1, 0:-1]
    # print("X1: ", X1.shape)
    # print("X1: ", X1)
    # train va du doan
    try:
        if phan_phoi == "Multinomial":
            model = MultinomialNB()
        elif phan_phoi == "Bernoulli":
            model = BernoulliNB()
        else:
            model = GaussianNB()
        model.fit(X, y)
        # print("3")
        pkl_filename = name + "_naive.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        print("testing: ", testing)
        print("training: ", training)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing, train_size=training)
        print("11111111: ", len(X_train))
        y_pred = model.predict(X_test)
        # print("y_test: ", y_test)
        # print("y_pred: ", y_pred)
        score = accuracy_score(y_test, y_pred)
    except Exception as e:
        print("err: ", e)
        score = 0
        return score
    print("score: ", score)
    return score

def TrainID3(data, name, training, testing):
    features = data.iloc[:-1, 1:-1].values
    # print("features: ", features)
    label = data.iloc[:-1, -1].values


    decisionTree = tree.DecisionTreeClassifier(max_depth=None, criterion='entropy', class_weight=None)
    # print("label: ", label)
    # print(len(label))
    decisionTree = decisionTree.fit(features, label)

    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=testing, train_size=training)
    # print("y: ", y_test)
    # print(X_train)
    X_train, X_test, y_train, y_test = train_test_split(features, label)
    y_pred = decisionTree.predict(X_test)
    score = accuracy_score(y_test, y_pred)


    try:
        # print("1: ", len(X_test))
        y_pred = decisionTree.predict(X_test)
        pkl_filename = name + "_" + "id3" + ".pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(decisionTree, file)
        print("2")
        score = accuracy_score(y_test.astype('int'), y_pred)
    except Exception as e:
        print("err: ", e)
        return 0
    return score