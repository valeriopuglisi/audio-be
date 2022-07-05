import os
from cfg import PIPELINES_PATH, REPORTS_PATH
from flask_restful import Resource, Api, reqparse
from flask import send_from_directory
import yaml


class Reports(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):
        reports = os.listdir(REPORTS_PATH)
        reports_json_list = []
        for file in reports:
            if file.endswith(".yaml"):
                file_path = os.path.join(REPORTS_PATH, file)
                with open(file_path, 'r') as file_yaml:
                    report_json = yaml.safe_load(file_yaml)
                    report_name = file.split(".")[0]
                    report_date_d, report_date_m, report_date_y = report_name.split("__")[1].split("_")
                    report_date = report_date_y + "-" + report_date_m + "-" + report_date_d
                    report_time = report_name.split("__")[2].replace("_", ":")
                    report_datetime = report_date + "T" + report_time
                    report_json['report_name'] = report_name
                    report_json['report_datetime'] = report_datetime

                    reports_json_list.append(report_json)
                print(reports_json_list)
                print(os.path.join(file_path))
        return reports_json_list, 201


class Report(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, report_id):
        filename_path = os.path.join(REPORTS_PATH, report_id)
        print("==> report_id: {}".format(report_id))
        print("==> report_path: {}".format(filename_path))
        return send_from_directory(REPORTS_PATH, report_id, as_attachment=True)

