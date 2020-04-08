from flask_restful import reqparse, Resource
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from werkzeug.datastructures import FileStorage
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn import preprocessing
from sklearn import utils

class TrainKnn(Resource):
    def post(self):
        mm_scaler  = preprocessing.MinMaxScaler()
        le = LabelEncoder()
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                        type=FileStorage,
                        location='files',
                        required=True,
                        help='CSV file')
        parser.add_argument(
            "neighbour", type=str, help="ko duoc bo trong"
        )
        data = parser.parse_args()
        try:
            dataToTrain_2 = pd.read_csv(data["resource_csv"])
        except Exception as e: 
            print("eeee: ", e)
            return {
                "msg": "Bad request1"
            }, 400
        if data["neighbour"] == "":
            neighbour = 3
        else:
            neighbour = int(data["neighbour"])
        # loc du lieu

        features = dataToTrain_2.iloc[:-1, 1:-1].values
        print("features: ", features)
        label = dataToTrain_2.iloc[:-1, -1].values
        # X_train, X_test, y_train, y_test = train_test_split(features, label)

        classifier = KNeighborsClassifier(n_neighbors=neighbour)
        # print("label: ", label)
        # print(len(label))
        
        X_train, X_test, y_train, y_test = train_test_split(features, label)
        print("y: ", y_test)
        # print(X_train)
        classifier.fit(features, label.astype('int'))
        print("4")
        try:
            print("1: ", len(X_test))
            y_pred = classifier.predict(X_test)
            pkl_filename = "knn.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(classifier, file)
            print("2")
            score = accuracy_score(y_test.astype('int'), y_pred)
        except Exception as e:
            print("err: ", e)
            return {
                "msg": "bad request",
            }, 400
        return {
            "score": score,
        }, 200



        
class Knn(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('Age',
                        type=float,
                        required=True,
                        help='tuoi')
        parser.add_argument('Gender',
                        type=float,
                        required=True,
                        help='nam hoacc nu')
        parser.add_argument('Education',
                        type=float,
                        required=True,
                        help='hoc van')
        parser.add_argument('Mariage',
                        type=float,
                        required=True,
                        help='tinh tran hon nhan')
        parser.add_argument('Properites',
                        type=float,
                        required=True,
                        help='gia canh')
        parser.add_argument('Salary',
                        type=float,
                        required=True,
                        help='luong')
        parser.add_argument('TimeToPay',
                        type=float,
                        required=True,
                        help='thoi gian tra')
        parser.add_argument('Debt',
                        type=float,
                        required=True,
                        help='so tien no')
        data = parser.parse_args()
        filename = "knn.pkl"
        try: 
            model = pickle.load(open(filename, 'rb'))
            test = [[data["Age"],data["Gender"],data["Education"],data["Mariage"],data["Properites"],
                                        data["Salary"],data["Debt"],data["TimeToPay"]]]
            print("test: ", test)
            predicted = model.predict([[data["Age"],data["Gender"],data["Education"],data["Mariage"],data["Properites"],
                                        data["Salary"],data["Debt"],data["TimeToPay"]]]) 

            print("3: ", predicted[0])
        except Exception as e:
            print("err: ", e)
            return {
                "msg": "bad request"
            }, 400
        return {
            "msg": predicted[0]
        }, 200
    


