import os
import subprocess
from deep_learning_dict_datasets import Datasets
import pandas as pd
import pathlib

path_to_ffmpeg_exe = "/usr/local/bin/ffmpeg"


def convert_common_voice_mp3_to_wav(task, dataset, input_dir, output_dir):
    # convert wav to mp3
    files_list = []
    # os.walk(dataset_path)
    test_table = pd.read_table(Datasets[task][dataset]["test_file"])
    n_files_to_convert = test_table.shape[0]
    dataset_path = Datasets[task][dataset]["path"]
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


def convert_iemocap_to_iemocap_audio_csv_format():

    iemocap_root_path = "data_8T/datasets/audio/IEMOCAP_full_release/"
    iemocap_audio_path = os.path.join(iemocap_root_path, "iemocap_audio_dataset.csv" )
    iemocap_dataset_session1_path = os.path.join(iemocap_root_path, "Session1")
    iemocap_dataset_session2_path = os.path.join(iemocap_root_path, "Session2")
    iemocap_dataset_session3_path = os.path.join(iemocap_root_path, "Session3")
    iemocap_dataset_session4_path = os.path.join(iemocap_root_path, "Session4")
    iemocap_dataset_session5_path = os.path.join(iemocap_root_path, "Session5")

    iemocap_session_paths = [
        iemocap_dataset_session1_path,
        iemocap_dataset_session2_path,
        iemocap_dataset_session3_path,
        iemocap_dataset_session4_path,
        iemocap_dataset_session5_path,
        ]


    iemocap_audio_dictionary = {
        'session':[],
        'audio_path': [],
        'gender':[],
        'emotion_evaluation_1': [],
        'emotion_evaluation_2': [],
        'emotion_evaluation_3': [],
        'emotion_evaluation_4': [],
        }

    for session_path in iemocap_session_paths:
        print("=====> ", session_path)
        session_dialog_path = os.path.join(session_path, "dialog", "EmoEvaluation")
        session_sentences_path = os.path.join(session_path, "sentences", "wav")
        for filename in os.listdir(session_dialog_path):
            file_path = os.path.join(session_dialog_path, filename)
            if not filename.startswith("._") and filename.endswith(".txt") and os.path.isfile(file_path):
                filename_no_ext = filename.split(".")[0]
                session_impro_sentence_dir_path = os.path.join(session_sentences_path, filename_no_ext)
                print("==> file_path:{}".format(file_path))
                print("==> filename:{}".format(filename))
                print("==> session_impro_sentence_path:{}".format(session_impro_sentence_dir_path))
                
                with open(file_path, "r") as file: # Use file to refer to the file object
                    lines = file.readlines()
                    for i,line in enumerate(lines):                      
                        if line.startswith("["):
                            print(i, line)
                            wav_audio_filename = line.split("\t")[1]
                            speaker_gender = wav_audio_filename.split("_")[-1][0]
                            print("speaker_gender:", speaker_gender)
                            wav_audio_emotion = line.split("\t")[2]
                            session_impro_sentence_path = os.path.join(session_impro_sentence_dir_path, wav_audio_filename)
                            iemocap_audio_dictionary['gender'].append(speaker_gender) 
                            iemocap_audio_dictionary['session'].append(session_path.split("/")[-1]) 
                            iemocap_audio_dictionary['audio_path'].append(session_impro_sentence_path) 
                            iemocap_audio_dictionary['emotion_evaluation_1'].append(wav_audio_emotion)
                            iemocap_audio_dictionary['emotion_evaluation_2'].append(lines[i+1].split("\t")[1][:-1])
                            iemocap_audio_dictionary['emotion_evaluation_3'].append(lines[i+2].split("\t")[1][:-1])
                            iemocap_audio_dictionary['emotion_evaluation_4'].append(lines[i+3].split("\t")[1][:-1])
                            # data_8T/datasets/audio/IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav
                            print("=> session_impro_sentence_path:{} - Emotion:{}".format(session_impro_sentence_path+".wav", wav_audio_emotion))
                            print()

    df = pd.DataFrame(iemocap_audio_dictionary)
    df.to_csv(iemocap_audio_path, index=True)           

        

# convert_iemocap_to_iemocap_audio_csv_format()

# _task = "Automatic Speech Recognition"
# _datasets = ["CommonVoice-DE-9.0",
#             "CommonVoice-ES-9.0",
#             "CommonVoice-EN-9.0",
#             "CommonVoice-IT-9.0",
#             "CommonVoice-FR-9.0",
#             "CommonVoice-DE-10.0",
#             "CommonVoice-ES-10.0",
#             "CommonVoice-EN-10.0",
#             "CommonVoice-IT-10.0",
#             "CommonVoice-FR-10.0"]

# _input_dir = "clips"
# _output_dir = "wavs"

# for _dataset in _datasets:
#     convert_common_voice_mp3_to_wav(_task, _dataset, _input_dir, _output_dir)