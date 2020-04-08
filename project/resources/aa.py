from flask_restful import Resource
from flask import request, Response, g
from src import server, cloud_camera
from src.modules import cores
from src.model.report import ReportModel
from src.model.history import HistoryModel, date_to_timestamp_2, get_day_from_month
from src.model.face import FaceModel
from src.model.count import CountModel
from src.model.camera import CameraModel
import cv2
from src.model import serializable

class Report(Resource):
    @server.login_required
    def get(self):
        try:
            id_user = g.user_id
            id_persons = []
            map_person = {}

            persons = FaceModel().find({'id_user': id_user})
            for person in persons:
                id_persons.append(person['id'])
                map_person[person['id']] = {'name': person['name'], 'code': person['code']}

            response = []
            reports = ReportModel().find({'id_person': {'$in': id_persons}})
            
            for r in reports:
                r = serializable(r)
                response.append({'name': map_person[r['id_person']]['name'], 'id_person': r['id_person'], 'code': map_person[r['id_person']]['code'], 'created_at': r['created_at']})

            return server.success(data={'reports': response})
        except Exception as e:
            raise Exception(e)

class ReportCountingByDay(Resource):
    @server.login_required
    def get(self, cam_id, date):
        print("date: ", date)
        try:
            report = ReportModel().find_by_date(cam_id = cam_id, date = date)
            if report is None:
                return server.success(data={'in': 0,
                                'out': 0})
            return server.success(data={'in': report["in"],
                            'out': report["out"]})
        except Exception as e:
            print("err: ", e)
            return server.bad_request()


class ReportCountingFromDayToDay(Resource):
    @server.login_required
    def get(self, cam_id, date_start, date_end):
        print("date_start: ", date_start)
        print("date_end: ", date_end)
        start, end = date_to_timestamp_2(date_start=date_start, date_end=date_end)
        data = CountModel().Count_movement_by_days(cam_id, start, end)
        print("data: ", data)
        return server.success(data=data)

class ReportCountingFromMonth(Resource):
    @server.login_required
    def get(self, cam_id, month, year):
        print("month: ", month)
        print("year: ", year)

        days = get_day_from_month(month, year)
        print("days")
        date_start = str(year) + '-' + str(month) + '-1' 
        date_end = str(year) + '-' + str(month) + '-' + str(days) 
        start, end = date_to_timestamp_2(date_start=date_start, date_end=date_end)

        data = CountModel().Count_movement_by_days(cam_id, start, end)
        print("data: ", data)
        return server.success(data=data)



class ReportDayByGroup(Resource):
    @server.login_required
    def get(self, group_id, date_start):
        try:
            start, end = date_to_timestamp_2(date_start = date_start, date_end="")
            if start == 0 or end == 0:
                return server.bad_request
            
            response, status_code = cloud_camera.get_list_camera_in_group(
                group_id=group_id, token=request.headers.get("Authorization")
            )
            if status_code == 500:
                raise Exception("call cloud camera server error" + response)
            elif status_code != 200:
                return server.custom(code=status_code, message=response)
            list_camera = response["cameras"]
            list_id = []
            for camera in list_camera:
                list_id.append(camera["id"])
            print("list_id: ", list_id)
            emotion = HistoryModel().Count_emotion_by_camID_with_specified_date(list_id, date_start, "")
            age_gender = HistoryModel().Count_gender_age_by_camID_with_specified_date(list_id, date_start, "")
            print("3")
            customer = HistoryModel().Count_visitor_by_camID_with_specified_date(list_id, date_start, "")
            print("4")
            customer_per_hour = HistoryModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start, "")

            # counting
            # visit = ReportModel().count_in_out_by_date(date = date_start, list_camID = list_id)
            visit_per_hour = CountModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start = date_start, date_end="")
            
                
            return {
                    "emotion": emotion,
                    "customer": customer,
                    "customer_per_hour": customer_per_hour,
                    "age_gender": age_gender,

                    "visit_per_hour": visit_per_hour,
                }
        except Exception as e:
            wrap_error(e)
            return server.bad_request

class ReportDaysByGroup(Resource):
    @server.login_required
    def get(self, group_id, date_start, date_end):
        try:
            response, status_code = cloud_camera.get_list_camera_in_group(
                group_id=group_id, token=request.headers.get("Authorization")
            )
            start, end = date_to_timestamp_2(date_start = date_start, date_end="")
            if start == 0 or end == 0:
                return server.bad_request

            if status_code == 500:
                raise Exception("call cloud camera server error" + response)
            elif status_code != 200:
                return server.custom(code=status_code, message=response)
            list_camera = response["cameras"]
            list_id = []
            for camera in list_camera:
                list_id.append(camera["id"])

            emotion = HistoryModel().Count_emotion_by_camID_with_specified_date(list_id, date_start, date_end)
            age_gender = HistoryModel().Count_gender_age_by_camID_with_specified_date(list_id, date_start, date_end)
            customer = HistoryModel().Count_visitor_by_camID_with_specified_date(list_id, date_start, date_end)
            customer_per_hour = HistoryModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start, date_end)

            # counting
            visit_per_day = ReportModel().count_in_out_by_dates(date_start=date_start, date_end=date_end, list_camID = list_id)
            
            visit_per_hour = CountModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start = date_start, date_end=date_end)
                
            return {
                    "emotion": emotion,
                    "customer": customer,
                    "customer_per_hour": customer_per_hour,
                    "age_gender": age_gender,
                    
                    "visit_per_day": visit_per_day,
                    "visit_per_hour": visit_per_hour,
                }
        except Exception as e:
            wrap_error(e)
            return server.bad_request

class ReportFromMonthByGroup(Resource):
    @server.login_required
    def get(self, group_id, month, year):
        try:
            days = 0
            if group_id is None or month is None or year is None:
                return server.bad_request
            
            days = get_day_from_month(month, year)
            date_start = str(year) + '-' + str(month) + '-1' 
            date_end = str(year) + '-' + str(month) + '-' + str(days) 

            start, end = date_to_timestamp_2(date_start = date_start, date_end="")
            if start == 0 or end == 0:
                return server.bad_request

            response, status_code = cloud_camera.get_list_camera_in_group(
                group_id=group_id, token=request.headers.get("Authorization")
            )
            if status_code == 500:
                raise Exception("call cloud camera server error" + response)
            elif status_code != 200:
                return server.custom(code=status_code, message=response)
            list_camera = response["cameras"]
            list_id = []

            for camera in list_camera:
                list_id.append(camera["id"])
            print("1")
            emotion = HistoryModel().Count_emotion_by_camID_with_specified_date(list_id, date_start, date_end)
            print("2")
            age_gender = HistoryModel().Count_gender_age_by_camID_with_specified_date(list_id, date_start, date_end)
            print("3")
            if month == "02":
                customer = HistoryModel().Count_visitor_by_camID_with_specified_date(list_id, date_start, date_end, february=True)
            else:
                customer = HistoryModel().Count_visitor_by_camID_with_specified_date(list_id, date_start, date_end, february=False)
            print("4")
            customer_per_hour = HistoryModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start, date_end)
            print("5")
            customer_day_by_day = HistoryModel().Count_customer_day_by_day(list_id, month, year)


            # couting
            days = get_day_from_month(month, year)
            date_start = str(year) + '-' + str(month) + '-1' 
            date_end = str(year) + '-' + str(month) + '-' + str(days) 
            start, end = date_to_timestamp_2(date_start=date_start, date_end=date_end)

            data = CountModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start, date_end)
            visit_per_day = ReportModel().count_in_out_by_dates(date_start=date_start, date_end=date_end, list_camID = list_id)
            visit_per_hour = CountModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start = date_start, date_end=date_end)
            return {
                "emotion": emotion,
                "age_gender": age_gender,
                "customer": customer,
                "customer_per_hour": customer_per_hour,
                "customer_day_by_day": customer_day_by_day,
                "couting": data,
                "visit_per_day": visit_per_day,
                "visit_per_hour": visit_per_hour
            }
        except Exception as e:
            wrap_error(e)
            return server.bad_request

class OverallReportByDay(Resource):
    @server.login_required
    def get(self, userID, date_start):
        try:
            userid = g.user_id 
            if userID != userid:
                return server.bad_request
            list_id = CameraModel().find_and_return_listID({'id_user': userID})
            print("list_id: ", list_id)
            response_group, status_code = cloud_camera.get_group(
                token=request.headers.get("Authorization")
            )
            print("response_group: ", response_group)
            list_group = response_group["groups"]
            print("list_group: ", list_group)
            start, end = date_to_timestamp_2(date_start = date_start, date_end="")
            if start == 0 or end == 0:
                return server.bad_request
            if status_code == 500:
                raise Exception("call cloud camera server error" + response_group)
            elif status_code != 200:
                return server.custom(code=status_code, message=response_group)

            emotion = HistoryModel().Count_emotion_by_camID_with_specified_date(list_id, date_start, "")
            age_gender = HistoryModel().Count_gender_age_by_camID_with_specified_date(list_id, date_start, "")
            customer = HistoryModel().Count_visitor_by_camID_with_specified_date(list_id, date_start, "")
            customer_per_hour = HistoryModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start, "")
            visit_per_hour_total = []
            for group in list_group:
                response, status_code = cloud_camera.get_list_camera_in_group(
                group_id=group["id"], token=request.headers.get("Authorization")
                )
                if status_code == 500:
                    raise Exception("call cloud camera server error" + response)
                elif status_code != 200:
                    return server.custom(code=status_code, message=response)
                list_camera = response["cameras"]
                list_id = []

                for camera in list_camera:
                    list_id.append(camera["id"])

                # counting
                # visit = ReportModel().count_in_out_by_date(date = date_start, list_camID = list_id)
                visit_per_hour = CountModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start = date_start, date_end="")

                a = {
                     "name": group["name"],
                     "number": visit_per_hour}
                visit_per_hour_total.append(a)
                    
            return {
                    "emotion": emotion,
                    "customer": customer,
                    "customer_per_hour": customer_per_hour,
                    "age_gender": age_gender,
                    "visit_per_hour": visit_per_hour_total,
                }
        except Exception as e:
            wrap_error(e)
            return server.bad_request

class OverallReportByDays(Resource):
    @server.login_required
    def get(self, userID, date_start, date_end):
        try:
            userid = g.user_id 
            if userID != userid:
                return server.bad_request
            list_id = CameraModel().find_and_return_listID({'id_user': userID})
            print("list_id: ", list_id)
            response_group, status_code = cloud_camera.get_group(
                token=request.headers.get("Authorization")
            )
            print("response_group: ", response_group)
            list_group = response_group["groups"]
            print("list_group: ", list_group)
            start, end = date_to_timestamp_2(date_start = date_start, date_end=date_end)
            if start == 0 or end == 0:
                return server.bad_request
            if status_code == 500:
                raise Exception("call cloud camera server error" + response_group)
            elif status_code != 200:
                return server.custom(code=status_code, message=response_group)

            emotion = HistoryModel().Count_emotion_by_camID_with_specified_date(list_id, date_start, date_end)
            age_gender = HistoryModel().Count_gender_age_by_camID_with_specified_date(list_id, date_start, date_end)
            customer = HistoryModel().Count_visitor_by_camID_with_specified_date(list_id, date_start, date_end)
            customer_per_hour = HistoryModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start, date_end)
            visit_per_hour_total = []
            visit_per_day_total = []
            for group in list_group:
                response, status_code = cloud_camera.get_list_camera_in_group(
                group_id=group["id"], token=request.headers.get("Authorization")
                )
                if status_code == 500:
                    raise Exception("call cloud camera server error" + response)
                elif status_code != 200:
                    return server.custom(code=status_code, message=response)
                list_camera = response["cameras"]
                list_id = []

                for camera in list_camera:
                    list_id.append(camera["id"])

                # counting
                # visit = ReportModel().count_in_out_by_date(date = date_start, list_camID = list_id)
                visit_per_day = ReportModel().count_in_out_by_dates(date_start=date_start, date_end=date_end, list_camID = list_id)
                visit_per_hour = CountModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start = date_start, date_end=date_end)

                a = {
                     "name": group["name"],
                     "number_hour": visit_per_hour,
                     "number_day": visit_per_day}
                visit_per_hour_total.append(a)
                    
            return {
                    "emotion": emotion,
                    "customer": customer,
                    "customer_per_hour": customer_per_hour,
                    "age_gender": age_gender,
                    "visit_per_hour": visit_per_hour_total,
                }
        except Exception as e:
            wrap_error(e)
            return server.bad_request

class OverallReportByMonth(Resource):
    @server.login_required
    def get(self, userID, month, year):
        try:
            days = get_day_from_month(month, year)
            date_start = str(year) + '-' + str(month) + '-1' 
            date_end = str(year) + '-' + str(month) + '-' + str(days) 

            userid = g.user_id 
            if userID != userid:
                return server.bad_request
            list_id = CameraModel().find_and_return_listID({'id_user': userID})
            print("list_id: ", list_id)
            response_group, status_code = cloud_camera.get_group(
                token=request.headers.get("Authorization")
            )
            print("response_group: ", response_group)
            list_group = response_group["groups"]
            print("list_group: ", list_group)
            start, end = date_to_timestamp_2(date_start = date_start, date_end=date_end)
            if start == 0 or end == 0:
                return server.bad_request
            if status_code == 500:
                raise Exception("call cloud camera server error" + response_group)
            elif status_code != 200:
                return server.custom(code=status_code, message=response_group)

            emotion = HistoryModel().Count_emotion_by_camID_with_specified_date(list_id, date_start, date_end)
            age_gender = HistoryModel().Count_gender_age_by_camID_with_specified_date(list_id, date_start, date_end)
            customer = HistoryModel().Count_visitor_by_camID_with_specified_date(list_id, date_start, date_end)
            customer_per_hour = HistoryModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start, date_end)
            visit_per_hour_total = []
            for group in list_group:
                response, status_code = cloud_camera.get_list_camera_in_group(
                group_id=group["id"], token=request.headers.get("Authorization")
                )
                if status_code == 500:
                    raise Exception("call cloud camera server error" + response)
                elif status_code != 200:
                    return server.custom(code=status_code, message=response)
                list_camera = response["cameras"]
                list_id = []

                for camera in list_camera:
                    list_id.append(camera["id"])

                # counting
                # visit = ReportModel().count_in_out_by_date(date = date_start, list_camID = list_id)
                visit_per_hour = CountModel().Count_visitor_by_camID_and_hour_with_specified_date(list_id, date_start = date_start, date_end=date_end)
                visit_per_day = ReportModel().count_in_out_by_dates(date_start=date_start, date_end=date_end, list_camID = list_id)
                a = {
                     "name": group["name"],
                     "number_hour": visit_per_hour,
                     "number_day": visit_per_day}
                visit_per_hour_total.append(a)

                
            return {
                    "emotion": emotion,
                    "customer": customer,
                    "customer_per_hour": customer_per_hour,
                    "age_gender": age_gender,
                    "visit_per_hour": visit_per_hour_total,
                }
                    
        except Exception as e:
            wrap_error(e)
            return server.bad_request
 