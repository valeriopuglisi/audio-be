from flask_restful import Resource, Api, reqparse
from flask import request
import werkzeug
from deep_learning_pipelines_features import *
from cfg import MEDIA_DIR


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
        pipelines_path = os.path.join(PIPELINES_PATH, 'pipelines.yaml')
        with open(pipelines_path, 'r') as pipelines:
            cur_yaml = yaml.safe_load(pipelines)
            print(cur_yaml)
        return cur_yaml, 201


class Pipeline(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, id):
        print("==> id:{}".format(id))
        pipelines_path = os.path.join(PIPELINES_PATH, 'pipelines.yaml')
        with open(pipelines_path, 'r') as pipelines:
            pipelines_yaml = yaml.safe_load(pipelines)
            pipeline = pipelines_yaml[int(id)]
            pipeline_name = pipeline['name']
            print(pipeline_name)
            pipeline_path = os.path.join(PIPELINES_PATH, pipeline_name + '.yml')
            with open(pipeline_path, 'r') as pipeline:
                pipeline_yaml = yaml.safe_load(pipeline)
                print(pipeline_yaml)
        return pipeline_yaml, 201

    def post(self, id):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_name = audiofile.filename
        audiofile_path = os.path.join(MEDIA_DIR, audiofile_name)
        print("audiofile_path : {} ".format(audiofile_path))
        print("audiofile_name : {} ".format(audiofile_name))
        print("pipeline_id : {} ".format(id))
        audiofile.save(audiofile_path)
        steps, pipeline = run_pipeline(audiofile_path=audiofile_path, pipeline_id=id)
        report_id = create_report(steps, pipeline)

        return [pipeline, report_id], 201

