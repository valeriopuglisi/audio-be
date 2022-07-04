import os
import subprocess
from deep_learning_dict_datasets import Datasets
import pandas as pd
import pathlib

path_to_ffmpeg_exe = "C:/ffmpeg/bin/ffmpeg.exe"


def convert_common_voice_mp3_to_wav(task, dataset, input_dir, output_dir):
    # convert wav to mp3
    files_list = []
    # os.walk(dataset_path)
    test_table = pd.read_table(Datasets[task][dataset]["test_file"])
    n_files_to_convert = test_table.shape[0]
    dataset_path = Datasets[task][dataset]["dataset_path"]
    input_dir = os.path.join(dataset_path, input_dir)
    output_dir = os.path.join(dataset_path, output_dir)
    if not os.path.isdir(output_dir):
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(test_table.iterrows()):
        print("format converting : {}/{}".format(i, n_files_to_convert))
        file_nm = row[1]["path"]
        reference = row[1]["sentence"]
        input_file = os.path.join(input_dir, file_nm)
        output_file = os.path.join(output_dir, str(file_nm.split(".")[0] + ".wav"))
        subprocess.call([path_to_ffmpeg_exe, '-i', input_file, output_file])
        # print("audiofile_path: {} - reference: {}".format(file_nm, reference))
        # print("input_file: {}".format(input_file))
        # print("output_file: {}".format(output_file))


# _task = "Automatic Speech Recognition"
# _dataset = "CommonVoice EN"
# _input_dir = "clips"
# _output_dir = "wavs"
# convert_common_voice_mp3_to_wav(_task, _dataset, _input_dir, _output_dir)
