# coding=utf-8
from model.score import Score
from model.train import Train
from utils.predict_processing import VerifyAndChangeData

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


class Load_Scrore_Table(Resource):
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        data = parser.parse_args()
        print("data: ", data)
        # preprocessing

        try:
            dataToPredict = pd.read_csv(data["resource_csv"])
        except Exception as e: 
            print("err: ", e)
            return {
                "msg": "Bad request"
            }, 400

        list_student = dataToPredict.iloc[:, :].values.tolist()
        print("type: ", type(list_student))
        header = list(dataToPredict.columns.values)
        print("list_student: ", type(list_student))
        print("header: ", type(header))
        return {
            "values": list_student[0],
            "header": header
        }, 200

class Preprocessing_Predict_Data(Resource):
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
                "msg": "Bad request"
            }, 400
        try:
                dataToPredict = pd.read_csv(data["resource_csv"])
        except: 
            return {
                "msg": "Bad request"
            }, 400
        result_list = dataToPredict.iloc[:, 1:].values.tolist()
        predict_subjects = list(dataToPredict.columns.values)
        predict_subjects = predict_subjects[1:]
        print("result_list: ", result_list[0])
        print("predict_subjects: ", predict_subjects)
        # kiểm tra và thay đổi dữ liệu theo form chuẩn
        try:
            predict_subjects, ordered_result_list, incompliance_subject, unecessary_subject = VerifyAndChangeData(khoa=data["khoa"],
                                                                                                                  thuat_toan=data["thuat_toan"],
                                                                                                                  predict_subjects=predict_subjects,
                                                                                                                  result_list=result_list[0])
        except Exception as e:
            print("err: ", e)
            return {
                "msg": "bad_request"
            }, 400                                                                                    
        return {
            "predict_subjects": predict_subjects, 
            "ordered_result_list": ordered_result_list, 
            "incompliance_subject": incompliance_subject, 
            "unecessary_subject": unecessary_subject
        }, 200


class PredictOverall(Resource):
    def post(self):
        print("1")
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

        if data["thuat_toan"] == "":
            return {
                "msg": "Bad request"
            }, 400
        try:
             dataToPredict = pd.read_csv(data["resource_csv"])
             dataToPredict = dataToPredict.fillna(value=0)
        except: 
            return {
                "msg": "Bad request"
            }, 400

        result_list = dataToPredict.iloc[:, 1:].values.tolist()
        predict_subjects = list(dataToPredict.columns.values)
        predict_subjects = predict_subjects[1:]
        # kiểm tra và thay đổi dữ liệu theo form chuẩn
        try:
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
            predict_result_toan_tin, score, recommend = transformed_mark_to_number_and_predict_knn(ordered_result_list, data["khoa"])
            print("toan_tin: ", predict_result_toan_tin)

        elif data["thuat_toan"] == "ID3":
            predict_result_toan_tin, score = transformed_mark_to_number_and_predict_id3(ordered_result_list, data["khoa"])
            print("toan_tin: ", predict_result_toan_tin[0][0])
            

        else: 
            predict_result_toan_tin, score = transformed_mark_to_number_and_predict_naive(ordered_result_list, data["khoa"])

            print("toan_tin: ", predict_result_toan_tin[0][0])
           
        # list_student_fewer = list(dataToPredict.iloc[1, 1:3].values)
        # print("list_student_fewer: ", list_student_fewer)
        # # sau có dữ liệu full thì bỏ dấu ngoặc ở dòng dưới đi
        # predict_result_1 = transformed_mark_to_number_and_predict_job([list_student_fewer])
        # print("predict_result_1: ", predict_result_1)

        print("toan_tin: ", predict_result_toan_tin)
        return {
                "toan_tin": predict_result_toan_tin,
                "score": score,
                # "recommend": recommend
            }, 200

def transformed_mark_to_number_and_predict_knn(list_student, khoa):
    filename = khoa + "_knn.pkl"
    model = pickle.load(open(filename, 'rb'))
    # load saved_score
    saved_score = Score.find_one(query={"train_model": khoa + "_knn"})
 
    print("list: ", list_student)
    neighbors_predict = model.kneighbors([list_student])

    neighbours = neighbors_predict[1][0].tolist()
    print("neighbour: ", neighbours)
    labels = get_label_for_knn(neighbors=neighbours, model_name=khoa + "_knn")
    print("label: ", labels)
    results, recommend = transform_label(labels=labels)
    print("result: ", results)
    print("recommend: ", recommend)
    
    return results, saved_score, recommend

def transformed_mark_to_number_and_predict_naive(list_student, khoa):
    filename = khoa + "_naive.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
    # load saved_score
    saved_score = Score.find_one(query={"train_model": khoa + "_naive"})
    print("list_student: ", list_student)
    mark_transformed = []
    # for mark in list_student:
    #     if mark >= 4 and mark < 5.5:
    #         mark_transformed.append(constant.DIEM_Y)
    #     elif mark >= 5.5 and mark < 6.5:
    #         mark_transformed.append(constant.DIEM_TB)
    #     elif mark >= 6.5 and mark <= 8:
    #         mark_transformed.append(constant.DIEM_K)
    #     elif mark >= 8 and mark <= 10:
    #         mark_transformed.append(constant.DIEM_G)
    #     elif mark == 0:
    #         mark_transformed.append(constant.KO_HOC)
    #     elif mark == -1:
    #         mark_transformed.append(constant.CT)
    #     else:
    #         print("lot: ", mark)
        # mark_transformed = np.array(mark_transformed)
    print("mark1: ", len(list_student))
    predicted = model.predict([list_student[:]])
    log_proba = model.predict_proba([list_student[:]])
    print("log_proba: ", log_proba)
    print("predicted: ", predicted)
    if predicted == [0]:
        result.append(["không ra được trường"])
    elif predicted == [1]:
        result.append(["giỏi"])
    elif predicted == [2]:
        result.append(["khá"])
    elif predicted == [3]:
        result.append(["trung bình"])
    elif predicted == [4]:
        result.append(["trung bình"])
    return result, saved_score

def transformed_mark_to_number_and_predict_id3(list_student, khoa):
    filename = khoa + "_id3.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
    # load saved_score
    saved_score = Score.find_one(query={"train_model": khoa + "_id3"})

    for student in list_student:
        mark_transformed = []
        for mark in student:
            if mark >= 4 and mark < 5.5:
                mark_transformed.append(constant.DIEM_Y)
            elif mark >= 5.5 and mark < 6.5:
                mark_transformed.append(constant.DIEM_TB)
            elif mark >= 6.5 and mark <= 8:
                mark_transformed.append(constant.DIEM_K)
            elif mark >= 8 and mark <= 10:
                mark_transformed.append(constant.DIEM_G)
            elif mark == 0:
                mark_transformed.append(constant.KO_HOC)
            elif mark == -1:
                mark_transformed.append(constant.CT)
            else:
                print("lot: ", mark)
        # mark_transformed = np.array(mark_transformed)
        predicted = model.predict([mark_transformed])
        if predicted == [0]:
            result.append(["không ra được trường"])
        elif predicted == [1]:
            result.append(["giỏi"])
        elif predicted == [2]:
            result.append(["khá"])
        elif predicted == [3]:
            result.append(["trung bình"])
        elif predicted == [4]:
            result.append(["trung bình"])
    
    return result, saved_score


def get_label_for_knn(neighbors, model_name):
    labels = []
    for neighbor in neighbors:
        data = {
            "position": int(neighbor),
            "identify": model_name
        }
        label = Train.find_one(query=data)
        labels.append(label)
    return labels

def get_label_for_naive(proba):
    labels = []
    copy_proba = proba
    sorted(copy_proba)
    top_3_label = copy_proba[:3]

    for x in top_3_label:
        label = proba.index(x)
        labels.append(label)
    
    print("label: ", labels)
    return labels

def transform_label(labels):
    results = []
    recommend = []
    max = 1
    for label in labels:
        nganh = ""
        number = int(label / 4)
        if number == 0:
            nganh = "TC "
        elif number == 1:
            nganh = "TI "
        elif number == 2:
            nganh = "TE "
        else:
            nganh = "TM "

        xeploai = label % 4
        # if xeploai == 0:
        #     result = result + " với xếp loại trung bình"
        # elif xeploai == 1:
        #     result = result + "với xếp loại trung bình khá "
        # elif xeploai == 2:
        #     result = result + "với xếp loại khá "
        # else:
        #     result = result + "với xếp loại giỏi "

        if xeploai == 0:
            result = nganh + "voi xep loai trung binh"
        elif xeploai == 1:
            result = nganh + "voi xep loai trung binh kha"
        elif xeploai == 2:
            result = nganh + "voi xep loai kha "
        else:
            result = nganh + "voi xep loai gioi "
        
        if xeploai > max:
            if len(recommend) > 0:
                recommend.pop(0)
            recommend.append(nganh)
            max = xeploai
        elif xeploai == max:
            recommend.append(nganh)
        
        results.append(result)
    return results, recommend