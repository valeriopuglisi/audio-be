from flask_restful import Resource, Api, reqparse
import json
from flask import request
import yaml
import pathlib
import os

PIPELINES_PATH = os.path.join(os.getcwd(), "pipelines")
pathlib.Path(PIPELINES_PATH).mkdir(parents=True, exist_ok=True)


class SavePipeline(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument('pipeline')
        json_data = request.get_json()
        print(type(json_data))
        print(json_data)
        pipeline_path = os.path.join(PIPELINES_PATH, json_data['name']+".yml")
        pipeline_name = json_data['name']
        pipeline_description = json_data['notes']
        pipelines_path = os.path.join(PIPELINES_PATH, 'pipelines.yaml')
        new_yaml_data_dict = {
            len(os.listdir(PIPELINES_PATH)) -1 : {
                'name': pipeline_name,
                'description': pipeline_description
            }
        }
        with open(pipeline_path, 'w') as outfile:
            yaml.dump(json_data, outfile, default_flow_style=False)

        with open(pipelines_path, 'r') as pipelines:
            cur_yaml = yaml.safe_load(pipelines)
            cur_yaml.update(new_yaml_data_dict)
            print(cur_yaml)

        with open(pipelines_path, 'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile)  # Also note the safe_dump

        return "Pipeline Stored !", 201


class Pipelines(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):

        pipelines = os.listdir(PIPELINES_PATH)
        return pipelines, 201