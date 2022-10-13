import os
import glob
import pandas as pd


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
                        wav_audio_emotion = line.split("\t")[2]
                        session_impro_sentence_path = os.path.join(session_impro_sentence_dir_path, wav_audio_filename)
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

    

    