import os
from cfg import PIPELINES_PATH, REPORTS_PATH
from flask_restful import Resource, Api, reqparse
from flask import send_from_directory


class Reports(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):
        reports = os.listdir(REPORTS_PATH)
        return reports, 201


class Report(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, report_id):
        filename_path = os.path.join(REPORTS_PATH, report_id)
        print("==> report_id: {}".format(report_id))
        print("==> report_path: {}".format(filename_path))
        return send_from_directory(REPORTS_PATH, report_id, as_attachment=True)

