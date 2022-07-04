import json
from flask_restful import Resource, reqparse
from deep_learning_dict_metrics import Metrics


class ApiMetricsList(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):
        metrics_list = json.dumps(Metrics)
        return metrics_list

