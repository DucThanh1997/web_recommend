# coding=utf-8
from model.score import Score
from model.train import Train
from utils.predict_processing import VerifyAndChangeData

from flask_restful import reqparse, Resource
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from common import constant
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
        header = list(dataToPredict.columns.values)
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
        # try:
        #     predict_subjects, ordered_result_list, incompliance_subject, unecessary_subject = VerifyAndChangeData(khoa=data["khoa"],
        #                                                                                                           thuat_toan=data["thuat_toan"],
        #                                                                                                           predict_subjects=predict_subjects,
        #                                                                                                           result_list=result_list[0])
        #     print("predict_subjects123: ", predict_subjects)
        #     print("ordered_result_list: ", ordered_result_list)
        # except Exception as e:
        #     print("err: ", e)
        #     return {
        #         "msg": "bad_request"
        #     }, 400                                                                                    
        # return {
        #     "predict_subjects": predict_subjects, 
        #     "ordered_result_list": ordered_result_list, 
        #     "incompliance_subject": incompliance_subject, 
        #     "unecessary_subject": unecessary_subject

        # }, 200
        return {
            "predict_subjects": predict_subjects, 
            "ordered_result_list": result_list[0], 
        }, 200

class PredictOverall(Resource):
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
        print("data: ", data["thuat_toan"])
        if data["thuat_toan"] == "knn":
            predict, score, recommend = transformed_mark_to_number_and_predict_knn(ordered_result_list, data["khoa"])

        elif data["thuat_toan"] == "ID3":
            predict, score, recommend = transformed_mark_to_number_and_predict_id3(ordered_result_list, data["khoa"])
            print("predict: ", predict)
            

        else: 
            predict, score, recommend = transformed_mark_to_number_and_predict_naive(ordered_result_list, data["khoa"])

            print("predict: ", predict)
           
        # list_student_fewer = list(dataToPredict.iloc[1, 1:3].values)
        # print("list_student_fewer: ", list_student_fewer)
        # # sau có dữ liệu full thì bỏ dấu ngoặc ở dòng dưới đi
        # predict_result_1 = transformed_mark_to_number_and_predict_job([list_student_fewer])
        # print("predict_result_1: ", predict_result_1)

        print("recommend: ", recommend)
        return {
                "toan_tin": predict[0:5],
                "score": score,
                "recommend": recommend
            }, 200

def transformed_mark_to_number_and_predict_knn(list_student, khoa):
    filename = khoa + "_knn.pkl"
    model = pickle.load(open(filename, 'rb'))
    # load saved_score
    saved_score = Score.find_one(query={"train_model": khoa + "_knn"})
    neighbors_predict = model.kneighbors([list_student])
    neighbours = neighbors_predict[1][0].tolist()
    print("neighbours: ", neighbours)
    labels = get_label_for_knn(neighbors=neighbours, model_name=khoa + "_knn")
    results, recommend = transform_label_knn(labels=labels, khoa=khoa)
    print("result: ", results)
    print("recommend: ", recommend)
    
    return results, saved_score, recommend

def transformed_mark_to_number_and_predict_naive(list_student, khoa):
    filename = khoa + "_naive.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
    # load saved_score
    saved_score = Score.find_one(query={"train_model": khoa + "_naive"})

    proba = model.predict_proba([list_student[:]])
    print("proba: ", proba * 100000000)
    print("predict123: ", model.predict([list_student[:]]))
    predicted, recommend = get_label_for_naive(list(proba[0]), khoa=khoa)
    # print("label: ", labels)
    print("predicted: ", predicted)
    return predicted, saved_score, recommend

def transformed_mark_to_number_and_predict_id3(list_student, khoa):
    filename = khoa + "_id3.pkl"
    model = pickle.load(open(filename, 'rb'))
    result = []
    # load saved_score
    saved_score = Score.find_one(query={"train_model": khoa + "_id3"})

    proba = model.predict_proba([list_student[:]])
    print("proba: ", proba * 100000000)
    print("predict123: ", model.predict([list_student[:]]))
    predicted, recommend = get_label_for_naive(list(proba[0]), khoa=khoa)
   
    # print("label: ", labels)
    print("predicted: ", predicted)
    return predicted, saved_score, recommend
    



def get_label_for_knn(neighbors, model_name):
    labels = []
    for neighbor in neighbors:
        data = {
            "position": int(neighbor),
            "identify": model_name
        }
        label = Train.find_one(query=data)
        labels.append(label)
    print("labels: ", labels)
    return labels

def get_label_for_naive(proba, khoa):
    labels = []
    copy_proba = proba.copy()
    print("copy_proba: ", len(copy_proba))
    copy_proba.sort(reverse=True)
    print("copy_proba: ", copy_proba)
    print("proba: ", proba)
    top_3_label = copy_proba[:3]
    print("top_3_label: ", top_3_label)
    max = 0
    for x in top_3_label:
        label = proba.index(x) + 1
        print("label: ", label)
        
        if khoa == "toan_tin":
            if label == 1:
                nganh = "TC với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 2:
                nganh = "TC với xếp loại giỏi"
                labels.append(nganh)
            elif label == 3:
                nganh = "TC với xếp loại khá"
                labels.append(nganh)
            elif label == 4:
                nganh = "TC với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 5:
                nganh = "TC với xếp loại trung bình"
                labels.append(nganh)
            elif label == 6:
                nganh = "TE với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 7:
                nganh = "TE với xếp loại giỏi"
                labels.append(nganh)
            elif label == 8:
                nganh = "TE với xếp loại khá"
                labels.append(nganh)
            elif label == 9:
                nganh = "TE với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 10:
                nganh = "TE với xếp loại trung bình"
                labels.append(nganh)
            elif label == 11:
                nganh = "TI với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 12:
                nganh = "TI với xếp loại giỏi"
                labels.append(nganh)
            elif label == 13:
                nganh = "TI với xếp loại khá"
                labels.append(nganh)
            elif label == 14:
                nganh = "TI với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 15:
                nganh = "TI với xếp loại trung bình"
                labels.append(nganh)
            elif label == 16:
                nganh = "TM với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 17:
                nganh = "TM với xếp loại giỏi"
                labels.append(nganh)
            elif label == 18:
                nganh = "TM với xếp loại khá"
                labels.append(nganh)
            elif label == 19:
                nganh = "TM với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 20:
                nganh = "TM với xếp loại trung bình"
                labels.append(nganh)

            print("nganh: ", nganh)
            if max <= (label % 5): 
                recommend = nganh[0:2]
        elif khoa == "kinh_te":
            if label == 1:
                nganh = "QA với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 2:
                nganh = "QA với xếp loại giỏi"
                labels.append(nganh)
            elif label == 3:
                nganh = "QA với xếp loại khá"
                labels.append(nganh)
            elif label == 4:
                nganh = "QA với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 5:
                nganh = "QA với xếp loại trung bình"
                labels.append(nganh)
            elif label == 6:
                nganh = "QB với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 7:
                nganh = "QB với xếp loại giỏi"
                labels.append(nganh)
            elif label == 8:
                nganh = "QB với xếp loại khá"
                labels.append(nganh)
            elif label == 9:
                nganh = "QB với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 10:
                nganh = "QB với xếp loại trung bình"
                labels.append(nganh)
            elif label == 11:
                nganh = "QE với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 12:
                nganh = "QE với xếp loại giỏi"
                labels.append(nganh)
            elif label == 13:
                nganh = "QE với xếp loại khá"
                labels.append(nganh)
            elif label == 14:
                nganh = "QE với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 15:
                nganh = "QE với xếp loại trung bình"
                labels.append(nganh)
            elif label == 16:
                nganh = "QM với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 17:
                nganh = "QM với xếp loại giỏi"
                labels.append(nganh)
            elif label == 18:
                nganh = "QM với xếp loại khá"
                labels.append(nganh)
            elif label == 19:
                nganh = "QM với xếp loại trung bình khá"
            elif label == 20:
                nganh = "QM với xếp loại trung bình"
                labels.append(nganh)
            if max <= (label % 5): 
                recommend = nganh[0:2]
        if khoa == "ngon_ngu":
            if label == 1:
                nganh = "NE với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 2:
                nganh = "NE với xếp loại giỏi"
                labels.append(nganh)
            elif label == 3:
                nganh = "NE với xếp loại khá"
                labels.append(nganh)
            elif label == 4:
                nganh = "NE với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 5:
                nganh = "NE với xếp loại trung bình"
                labels.append(nganh)
            elif label == 6:
                nganh = "NJ với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 7:
                nganh = "NJ với xếp loại giỏi"
                labels.append(nganh)
            elif label == 8:
                nganh = "NJ với xếp loại khá"
                labels.append(nganh)
            elif label == 9:
                nganh = "NJ với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 10:
                nganh = "NJ với xếp loại trung bình"
                labels.append(nganh)
            elif label == 11:
                nganh = "NK với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 12:
                nganh = "NK với xếp loại giỏi"
                labels.append(nganh)
            elif label == 13:
                nganh = "NK với xếp loại khá"
                labels.append(nganh)
            elif label == 14:
                nganh = "NK với xếp loại trung bình khá"
                labels.append(nganh)
            elif label == 15:
                nganh = "NK với xếp loại trung bình"
                labels.append(nganh)
            elif label == 16:
                nganh = "NZ với xếp loại xuất sắc"
                labels.append(nganh)
            elif label == 17:
                nganh = "NZ với xếp loại giỏi"
                labels.append(nganh)
            elif label == 18:
                nganh = "NZ với xếp loại khá"
                labels.append(nganh)
            elif label == 19:
                nganh = "NZ với xếp loại trung bình khá"
            elif label == 20:
                nganh = "NZ với xếp loại trung bình"
                labels.append(nganh)
            if max <= (label % 5): 
                recommend = nganh[0:2]
    print("label: ", labels)
    print("recommend: ", recommend)
    return labels, recommend

def transform_label_knn(labels, khoa):
    results = []
    recommend = []
    max = 2
    if khoa == "toan_tin":
        for label in labels[0:5]:
            nganh = ""
            number = int(label / 4)
            if number == 1:
                nganh = "TC "
            elif number == 2:
                nganh = "TE "
            elif number == 3:
                nganh = "TI "
            else:
                nganh = "TM "

            xeploai = label % 5
            if xeploai == 0:
                result = nganh + "với xếp loại xuất sắc"
            elif xeploai == 1:
                result = nganh + "với xếp loại giỏi"
            elif xeploai == 2:
                result = nganh + "với xếp loại khá"
            elif xeploai == 3:
                result = nganh + "với xếp loại trung bình khá"
            else:
                result = nganh + "với xếp loại trung bình"
            
            print("max: ", max)
            print("xeploai: ", xeploai)
            if xeploai <= max:
                recommend.append(nganh)
            
            results.append(result)
    elif khoa == "kinh_te":
        for label in labels[0:5]:
            nganh = ""
            number = int(label / 4)
            if number == 1:
                nganh = "QA "
            elif number == 2:
                nganh = "QB "
            elif number == 3:
                nganh = "QE "
            else:
                nganh = "QM "

            xeploai = label % 5
            if xeploai == 0:
                result = nganh + "với xếp loại xuất sắc"
            elif xeploai == 1:
                result = nganh + "với xếp loại giỏi"
            elif xeploai == 2:
                result = nganh + "với xếp loại khá"
            elif xeploai == 3:
                result = nganh + "với xếp loại trung bình khá"
            else:
                result = nganh + "với xếp loại trung bình"
            
            print("max: ", max)
            print("xeploai: ", xeploai)
            if xeploai <= max:
                recommend.append(nganh)
            
            results.append(result)
    else:
        for label in labels[0:5]:
            nganh = ""
            number = int(label / 4)
            if number == 1:
                nganh = "NE "
            elif number == 2:
                nganh = "NJ "
            elif number == 3:
                nganh = "NK "
            else:
                nganh = "NZ "

            xeploai = label % 5
            if xeploai == 0:
                result = nganh + "với xếp loại xuất sắc"
            elif xeploai == 1:
                result = nganh + "với xếp loại giỏi"
            elif xeploai == 2:
                result = nganh + "với xếp loại khá"
            elif xeploai == 3:
                result = nganh + "với xếp loại trung bình khá"
            else:
                result = nganh + "với xếp loại trung bình"
            print("max: ", max)
            print("xeploai: ", xeploai)
            if xeploai <= max:
                recommend.append(nganh)
            
            results.append(result)
    recommend = set(recommend)
    recommend = list(recommend)
    recommendd = ""
    for reco in recommend:
        if len(recommendd) < 1:
            recommendd = reco
        else:
            recommendd = recommendd + ", " + reco
    print("result: ", results)
    return results, recommendd