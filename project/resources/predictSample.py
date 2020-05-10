# coding=utf-8
from flask_restful import reqparse, Resource
from werkzeug.datastructures import FileStorage
import pandas as pd
import pickle


from utils.predict_processing import VerifyAndChangeData
from resources.predict_overall import (transformed_mark_to_number_and_predict_knn,
                                       transformed_mark_to_number_and_predict_id3, 
                                       transformed_mark_to_number_and_predict_naive)
from model.score import Score
                                      

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
        except: 
            return {
                "msg": "Bad request2"
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
        if data["thuat_toan"] == "Knn":
            predict, score, recommend = transformed_mark_to_number_and_predict_knn(ordered_result_list, data["khoa"])
            print("toan_tin: ", predict)

        elif data["thuat_toan"] == "ID3":
            predict, score = transformed_mark_to_number_and_predict_id3(ordered_result_list, data["khoa"])
            print("toan_tin: ", predict)
            

        else: 
            print("order: ", ordered_result_list)
            predict, score = transformed_mark_to_number_and_predict_naive_sample(ordered_result_list, data["khoa"])

           
        # list_student_fewer = list(dataToPredict.iloc[1, 1:3].values)
        # print("list_student_fewer: ", list_student_fewer)
        # # sau có dữ liệu full thì bỏ dấu ngoặc ở dòng dưới đi
        # predict_result_1 = transformed_mark_to_number_and_predict_job([list_student_fewer])
        # print("predict_result_1: ", predict_result_1)

        print("toan_tin: ", predict)
        return {
                "predict": predict,
                "score": score,
                # "recommend": recommend
            }, 200

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
    print("log_proba: ", log_proba)
    print("predicted: ", predicted)
    return predicted.tolist(), saved_score