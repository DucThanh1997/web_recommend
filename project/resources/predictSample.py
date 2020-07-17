# coding=utf-8
from __future__ import division
from flask_restful import reqparse, Resource
from werkzeug.datastructures import FileStorage
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

from utils.predict_processing import VerifyAndChangeData
from utils.train_sample_processing import Processing_data_knn, checktype
from resources.predict_overall import (transformed_mark_to_number_and_predict_knn,
                                       transformed_mark_to_number_and_predict_id3, 
                                       transformed_mark_to_number_and_predict_naive)
from model.score import Score
from model.header import Header
                                      

class PredictSample(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                            type=FileStorage,
                            location='files',
                            required=True,
                            help='CSV file')
        parser.add_argument('thuat_toan',
                            type=str, 
                            help="ko duoc bo trong")
        parser.add_argument('khoa',
                            type=str, 
                            help="ko duoc bo trong")
        data = parser.parse_args()
        print("data: ", data)
        if data["thuat_toan"] == "":
            return {
                "msg": "Bad request1"
            }, 400
        try:
            dataToPredict = pd.read_csv(data["resource_csv"])
            print("dataToPredict: ", dataToPredict)
        except: 
            return {
                "msg": "Bad request2"
            }, 400

        result_list = dataToPredict.iloc[:, 1:].values.tolist()
        predict_subjects = list(dataToPredict.columns.values)
        predict_subjects = predict_subjects[1:]
        # kiểm tra và thay đổi dữ liệu theo form chuẩn
        try:
            print("result: ", result_list)
            predict_subjects, ordered_result_list, _, _ = VerifyAndChangeData(khoa=data["khoa"],
                                                                              thuat_toan=data["thuat_toan"],
                                                                              predict_subjects=predict_subjects,
                                                                              result_list=result_list[0])
        except Exception as e:
            print("err: ", e)
            return {
                "msg": "wrong input"
            }, 400
        
        # B1 biến đổi dữ liệu và dự đoán tốt nghiệp đầu ra
        print("data: ", data["khoa"])
        if data["thuat_toan"] == "knn":
            predict, list_predict, score = transformed_mark_to_number_and_predict_knn_sample(ordered_result_list)
            print("score: ", score)
            return {
                    "predict": int(predict[0]),
                    "score": int(score)
                    # "recommend": recommend
            }, 200

        elif data["thuat_toan"] == "ID3":
            predict, score = transformed_mark_to_number_and_predict_id3_sample(ordered_result_list, data["khoa"])
            print("toan_tin: ", predict)
            if predict == 0:
                predict = "no"
            else:
                predict = "yes"
            return {
                "predict": predict,
                "score": int(score)
                # "recommend": recommend
            }, 200
            

        else: 
            print("order: ", ordered_result_list)
            predict, score = transformed_mark_to_number_and_predict_naive_sample(ordered_result_list, data["khoa"])
            print("predict: ", predict[0])
            print("score: ", score)
            return {
                    "predict": predict,
                    "score": int(score)
                    # "recommend": recommend
                }, 200
           
        # list_student_fewer = list(dataToPredict.iloc[1, 1:3].values)
        # print("list_student_fewer: ", list_student_fewer)
        # # sau có dữ liệu full thì bỏ dấu ngoặc ở dòng dưới đi
        # predict_result_1 = transformed_mark_to_number_and_predict_job([list_student_fewer])
        # print("predict_result_1: ", predict_result_1)



def transformed_mark_to_number_and_predict_naive_sample(list_student, khoa):
    filename = khoa + "_naive.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
    # load saved_score
    saved_score = Score.find_one(query={"train_model": khoa + "_naive"})
    print("list_student: ", list_student)
    mark_transformed = []
    
    print("mark1: ", len(list_student))
    predicted = model.predict([list_student[:]])
    log_proba = model.predict_proba([list_student[:]])

    headers_old = Header.find_one(query={
                "identify": "train_naive_sample"
            })
    if headers_old == -1:
        return predicted.tolist(), saved_score
    else:
        print("headers_old: ", headers_old)
        transformed_predict = headers_old[str(predicted.tolist()[0])]
        return transformed_predict, saved_score

    print("log_proba: ", log_proba)
    print("predicted: ", predicted)
    return predicted.tolist(), saved_score

def transformed_mark_to_number_and_predict_knn_sample(data):
    mm_scaler  = preprocessing.MinMaxScaler()
    filename = "sample_knn.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
    # load saved_score
    saved_score, maxx, minn = Score.find_one_sample(query={"train_model": "sample_knn"})
    mark_transformed = []
    print("data1: ", data)
    for index, value in enumerate(data):
        print("value: ", value)
        print("max: ", maxx[index])
        print("min: ", minn[index])
        value_transformed = (int(value) - minn[index]) / (maxx[index] - minn[index])
        print("value_transformed: ", value_transformed)
        mark_transformed.append(value_transformed)
    
    print("mark_transformed: ", mark_transformed)
    # a = data.shape
    # data_changed = Processing_data_knn(data=data, columns=a[1])
    neighbors_predict = model.kneighbors([mark_transformed])
    predict = model.predict([mark_transformed])

    return predict, neighbors_predict, saved_score


def transformed_mark_to_number_and_predict_id3_sample(list_student, khoa):
    print("list_student: ", list_student)
    filename = khoa + "_id3.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
    transform_data = []
    # load saved_score
    saved_score = Score.find_one(query={"train_model": khoa + "_id3"})
    transform_data = []
    for index, data in enumerate(list_student):
        if index == 0:
            if data == "sunny":
                print("sunny")
                transform_data.append(2)
            elif data == "overcast":
                print("overcast")
                transform_data.append(0)
            else:
                print("rainy")
                transform_data.append(1)
        elif index == 1:
            if data == "hot":
                print("hot")
                transform_data.append(1)
            elif data == "mild":
                print("mild")
                transform_data.append(2)
            else:
                print("cold")
                transform_data.append(0)
        elif index == 2:
            if data == "normal":
                print("normal")
                transform_data.append(1)
            else:
                print("high")
                transform_data.append(0)
        elif index == 3:
            if data == "weak":
                print("weak")
                transform_data.append(1)
            else:
                print("high")
                transform_data.append(0)
        
    print("transformed_data: ", transform_data)    
    proba = model.predict_proba([transform_data])
    predicted = model.predict([transform_data])
    # print("label: ", labels)
    print("predicted: ", predicted)
    return predicted, saved_score