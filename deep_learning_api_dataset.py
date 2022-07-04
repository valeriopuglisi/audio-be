import json
from pprint import pprint

import werkzeug
from flask import send_from_directory
from flask_restful import Resource, reqparse
from deep_learning_features_audio import *
from deep_learning_dict_datasets import Datasets


class ApiDatasetsList(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):
        datasets_list = json.dumps(Datasets)
        return datasets_list

