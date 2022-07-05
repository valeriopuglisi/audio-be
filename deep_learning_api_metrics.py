import json
from flask_restful import Resource, reqparse
from deep_learning_dict_metrics import Metrics
import os
from  cfg import *

class ApiMetricsList(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):
        metrics_list = json.dumps(Metrics)
        return metrics_list


class ApiEvaluationsList(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):
        reports = os.listdir(RESULTS_PATH)
        evaluation_json_list = []
        for file in reports:
            if file.endswith(".json"):
                file_path = os.path.join(RESULTS_PATH, file)
                with open(file_path, 'r') as file_json:
                    evaluation = json.load(file_json)
                    evaluation_json_list.append(evaluation)
        return evaluation_json_list

