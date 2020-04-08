# coding=utf-8

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
        print("2")
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

        list_student = list(dataToPredict.iloc[:, 1:].values)
        header = list(dataToPredict.columns.values)
        header = header[1:]
        print("3")
        # kiểm tra và thay đổi dữ liệu theo form chuẩn
        try:
            list_student_verify = VerifyAndChangeData(header_list=header, result_list=list_student)
        except Exception as e:
            print("err: ", e)
            return {
                "msg": "wrong input"
            }, 400
        print("list_student_verify: ", list_student_verify)
        ma_sv = list(dataToPredict.iloc[:, 0].values)
        # B1 biến đổi dữ liệu và dự đoán tốt nghiệp đầu ra
        print("data: ", data["thuat_toan"])
        if data["thuat_toan"] == "Knn":
            predict_result_toan_tin = transformed_mark_to_number_and_predict_results(list_student_verify, "Toan_tin")
            predict_result_kinh_te = transformed_mark_to_number_and_predict_results(list_student_verify, "kinh_te")
            predict_result_ngon_ngu = transformed_mark_to_number_and_predict_results(list_student_verify, "ngon_ngu")
            print("toan_tin: ", predict_result_toan_tin[0][0])
            print("kinh_te: ", predict_result_kinh_te[0][0])
            print("ngon_ngu: ", predict_result_ngon_ngu[0][0])
        else: 
            predict_result_toan_tin = transformed_mark_to_number_and_predict_results(list_student_verify, "Toan_tin")
            predict_result_kinh_te = transformed_mark_to_number_and_predict_results(list_student_verify, "kinh_te")
            predict_result_ngon_ngu = transformed_mark_to_number_and_predict_results(list_student_verify, "ngon_ngu")
        # list_student_fewer = list(dataToPredict.iloc[1, 1:3].values)
        # print("list_student_fewer: ", list_student_fewer)
        # # sau có dữ liệu full thì bỏ dấu ngoặc ở dòng dưới đi
        # predict_result_1 = transformed_mark_to_number_and_predict_job([list_student_fewer])
        # print("predict_result_1: ", predict_result_1)
        return {
                "toan_tin": predict_result_toan_tin[0][0],
                "kinh_te": predict_result_kinh_te[0][0],
                "ngon_ngu": predict_result_ngon_ngu[0][0],
            }, 200

def transformed_mark_to_number_and_predict_results(list_student, khoa):
    filename = khoa + "_knn.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
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
        result.append(predicted)
    
    return result

def transformed_mark_to_number_and_predict_job(list_student):
    filename = "naive.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
    for student in list_student:
        mark_transformed = []
        for mark in student:
            if mark >= 4 and mark < 7:
                mark_transformed.extend([1,0,0,0])
            elif mark >= 7 and mark < 8:
                mark_transformed.extend([0,1,0,0])
            elif mark >= 8 and mark < 9:
                mark_transformed.extend([0,0,1,0])
            elif mark >= 9 and mark <= 10:
                mark_transformed.extend([0,0,0,1])
            else:
                print("lot: ", mark)
        # mark_transformed = np.array(mark_transformed)
        print("mark1: ", [mark_transformed])
        predicted = model.predict([mark_transformed])
        result.append(predicted[0])
    return result

def VerifyAndChangeData(header_list, result_list):
    sort_list = ["CS100", "EC101", "EC102", "GE100", "GE101", "GE102", "GE141", "GE142", "GE143", "GE201", 
                 "GE202", "GE244", "GE245", "GE246", "GF101", "GF102", "GI101", "GI102", "GJ101", "GJ102", 
                 "GJ161", "GJ162", "GJ163", "GJ171", "GJ172", "GJ173", "GZ101", "GZ102", "GZ131", "GZ151A",
                 "GZ152A", "IS206", "MA101", "MA103", "ML111", "ML112", "ML202", "ML203", "NA151", "PG100",
                 "PG121", "PV101", "SH121", "SH131", "SO101"]


    
    # kiểm tra 2 list có bằng nhau ko
    if sort_list == header_list: 
        print("2 list giống nhau") 
        return result_list
    else : 
        print("bắt đầu quá trình chỉnh sửa") 
        correct_index = []
        # tìm thứ tự đúng
        for element, subject in enumerate(sort_list):
            if subject == header_list[element]:
                correct_index.append(element)
            else:
                for header_element, header_subject in enumerate(header_list):
                    if header_subject == subject:
                        correct_index.append(header_element)
                        break
                    else:
                        continue
        print(correct_index)

        for result in result_list:
            print("result: ", result)
            stop_change = []
            for number, mark in enumerate(result):
                if number == correct_index[number]:
                    continue
                else:
                    if number in stop_change:
                        continue
                    else:

                        temp = mark
                        temp1 = result[correct_index[number]]

                        result[number] = temp1,
                        print("result[number]: ", result[number])
                        
                        result[number] = int(result[number][0])
                        result[correct_index[number]] = temp

                        stop_change.append(correct_index[number])
                        
        return result_list