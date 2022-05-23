from zipfile import ZipFile
from datetime import datetime
import pathlib
import os
import yaml
from cfg import PIPELINES_PATH, REPORTS_PATH


def create_report(steps, _yaml):
    now = datetime.now()
    print("now =", now)
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d_%m_%Y__%H_%M_%S")
    report_id = "report__" + dt_string
    report_name = report_id + '.zip'
    report_path = os.path.join(REPORTS_PATH, report_name)
    pipeline_path = os.path.join(REPORTS_PATH, report_id + '.yaml')
    analysis_result_path = os.path.join(REPORTS_PATH, report_id + '_analysis_results.txt')

    # Create a ZipFile Object
    with ZipFile(report_path, 'w') as zipObj:
        with open(pipeline_path, 'w') as outfile:
            yaml.dump(_yaml, outfile, default_flow_style=False)
            zipObj.write(pipeline_path)
        # Add multiple files to the zip
        for i, step in enumerate(steps):
            print(step)
            print("step {} - input_file : {}".format(i, step['inputFilename']))
            zipObj.write(step['inputFilename'])
            for outputfile in step['outputFilenames']:
                print("step {} - outputfile: {}: ".format(i, outputfile))
                zipObj.write(outputfile)
            print("-----------------------------------------------------------------------")
            try:
                with open(analysis_result_path, 'a+', encoding='utf-8') as f:
                    input_filename = os.path.split(step["inputFilename"])[-1]
                    f.write("step {} - task: {} - input-file:{} - result: {}\n".format(i, step["task"], input_filename, step["analysisResult"]))
            except Exception as e:
                print("Exception :{}".format(e))
        zipObj.write(analysis_result_path)
        zipObj.close()
        return report_name
