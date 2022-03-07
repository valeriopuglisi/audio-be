from flask_restful import Resource, Api, reqparse
import json
from flask import request


class SavePipeline(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument('pipeline')
        json_data = request.get_json()
        print(type(json_data))
        print(json_data)
        # pipeline_dict = json.loads(pipeline)

        return "Pipeline Stored !", 201