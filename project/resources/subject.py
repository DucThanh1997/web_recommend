from __future__ import print_function 
import numpy as np 
import pandas as pd
import csv
import uuid
from flask_restful import reqparse, Resource 
from werkzeug.datastructures import FileStorage

from model.subject import Subject


class Subjects(Resource):
    def get(self):
        try:
            list_subjects = Subject().find(query={})
            return {
                "data": list_subjects
            }, 200
        except Exception as e:
            raise Exception(e)
    
    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('resource_csv',
                         type=FileStorage,
                         location='files',
                         required=True,
                         help='CSV file')
        data = parser.parse_args()
        try:
            dataToTrain = pd.read_csv(data["resource_csv"])
            subjects = dataToTrain.iloc[:, :].values
            subjects = list(subjects)
            print("data: ", type(subjects))
            i = 0
            for subject in subjects:
                Subject().insert(data= {"SubjectID":subject[0],
                                        "SubjectName": subject[1],
                                        "Nganh": subject[2]})
                i = i + 1
            # line_count += 1
            return {
                "msg": "okke",
                "number": i,
            }, 200
        except Exception as e: 
            print("e: ", e)
            return {
                "msg": "Bad request"
            }, 400