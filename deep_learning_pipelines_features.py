import yaml
from deep_learning_api_dict import *
from zipfile import ZipFile


def run_pipeline(audiofile_path, pipeline_id):
    print("==> id:{}".format(id))
    pipelines_path = os.path.join(PIPELINES_PATH, 'pipelines.yaml')
    with open(pipelines_path, 'r') as pipelines:
        pipelines_yaml = yaml.safe_load(pipelines)
        pipeline = pipelines_yaml[int(pipeline_id)]
        pipeline_name = pipeline['name']
        pipeline_path = os.path.join(PIPELINES_PATH, pipeline_name + '.yml')
        with open(pipeline_path, 'r') as pipeline:
            pipeline_yaml = yaml.safe_load(pipeline)
            steps = pipeline_yaml['steps']
            for i, step in enumerate(steps):
                print("========> STEP : ", i, step )
                api = step['api']
                task = step['task']
                system = step['system']
                dataset = step['dataset']
                performance = step['performance']
                input_file_info = step['inputFileId']
                output_file_infos = step['outputFileIds']
                print("- api: {} \n- task:{}\n- system:{}\n- dataset:{}\n- performance:{}".
                      format(api, task, system, dataset, performance))
                print("- input_file_id: ", input_file_info)
                print("- output_file_ids: ", output_file_infos)

                input_file_type = input_file_info.split("_")[0]
                input_file_step = int(input_file_info.split("_")[1])
                input_file_id = int(input_file_info.split("_")[2])
                print("- steps[input_file_step]: {}, ".format(steps[input_file_step]))

                # Select input_file for the first time
                if i == 0 :
                    step['inputFilename'] = audiofile_path

                # After first step check if the file in input for the i-th step
                # is an input file reletad to a previous step
                if input_file_type == 'input':
                    print("- steps[input_file_step]['inputFileId']: {}, ".
                          format(steps[input_file_step]['inputFileId']))
                    print("- steps[input_file_step]['inputFilename']: {}, ".
                          format(steps[input_file_step]['inputFilename']))
                    input_file_path = steps[input_file_step]['inputFilename']
                    print("- input_file_path: {}, ".format(input_file_path))
                    step['inputFilename'] = input_file_path

                # Or if the input file of the i-th step is an outputfile of a  previous step
                elif input_file_type == 'output':
                    print("- steps[input_file_step]['outputFileIds']: {}, ".
                          format(steps[input_file_step]['outputFileIds'][input_file_id]))

                    input_file_path = steps[input_file_step]['outputFilenames'][input_file_id]
                    step['inputFilename'] = input_file_path

                # Check what type of task I'm doing :
                # -- If it is a separation/enhancement task then put the results in outputFilenames
                #    because the result is the list of file names

                if step['task'] == "Speech Enhancement" or\
                        step['task'] == "Speech Separation" or \
                        step['task'] == "Audio Separation" or step['task'] == "Voice Activity Detection":
                    step['outputFilenames'] = AudioAnalysisAPI[api]['function'](audiofile_path=input_file_path)
                else:
                    step['analysisResult'] = AudioAnalysisAPI[api]['function'](audiofile_path=input_file_path)
                print("- step['inputFilename']: {}, ".format(step['inputFilename']))
                print("- step['outputFilenames']: {}, ".format(step['outputFilenames']))

            return pipeline_yaml

