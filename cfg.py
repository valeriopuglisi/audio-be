import os
import pathlib
import glob

MEDIA_DIR = os.path.join(os.getcwd(), "media")
VAD_CRDNN = os.path.join(os.getcwd(), "voice_activity_detection_crdnn")
AUDIO_SEPARATION_SEPFORMER_WHAMR = os.path.join(os.getcwd(), "audio_separation_sepformer_whamr")
AUDIO_SEPARATION_SEPFORMER_WHAM = os.path.join(os.getcwd(), "audio_separation_sepformer_wham")
ENHANCEMENT_SEPFORMER_WSJ0_DIR = os.path.join(os.getcwd(), "enhancement_sepformer_wsj0mix")
ENHANCEMENT_SEPFORMER_WHAMR_DIR = os.path.join(os.getcwd(), "enhancement_sepformer_whamr")
ENHANCEMENT_SEPFORMER_WHAMR_16k_DIR = os.path.join(os.getcwd(), "enhancement_sepformer_whamr-16k")
ENHANCEMENT_SEPFORMER_WHAM_DIR = os.path.join(os.getcwd(), "enhancement_sepformer_wham")
ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR = os.path.join(os.getcwd(), "enhancement_metricganplus_voicebank")
SEPARATION_SEPFORMER_WSJ3_DIR = os.path.join(os.getcwd(), "separation_sepformer_wsj03mix")
SEPARATION_SEPFORMER_WSJ2_DIR = os.path.join(os.getcwd(), "separation_sepformer_wsj02mix")
SEPARATION_WHAM_DIR = os.path.join(os.getcwd(), "separation_sepformer_wham")
SEPARATION_WHAMR_DIR = os.path.join(os.getcwd(), "separation_sepformer_whamr")

PIPELINES_PATH = os.path.join(os.getcwd(), "pipelines")
REPORTS_PATH = os.path.join(os.getcwd(), "reports")

pathlib.Path(REPORTS_PATH).mkdir(parents=True, exist_ok=True)

pathlib.Path(VAD_CRDNN).mkdir(parents=True, exist_ok=True)
pathlib.Path(REPORTS_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(PIPELINES_PATH).mkdir(parents=True, exist_ok=True)
pathlib.Path(AUDIO_SEPARATION_SEPFORMER_WHAMR).mkdir(parents=True, exist_ok=True)
pathlib.Path(AUDIO_SEPARATION_SEPFORMER_WHAM).mkdir(parents=True, exist_ok=True)
pathlib.Path(ENHANCEMENT_SEPFORMER_WSJ0_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(ENHANCEMENT_SEPFORMER_WHAM_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(ENHANCEMENT_SEPFORMER_WHAMR_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(ENHANCEMENT_SEPFORMER_WHAMR_16k_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(SEPARATION_WHAM_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(SEPARATION_WHAMR_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(SEPARATION_SEPFORMER_WSJ2_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(SEPARATION_SEPFORMER_WSJ3_DIR).mkdir(parents=True, exist_ok=True)
# print(MEDIA_DIR)
# print(SEPARATION_SEPFORMER_WSJ3_DIR)
media_files = glob.glob(os.path.join(os.getcwd(), "media", "*"))
# print(media_files)

# All files and directories ending with .txt and that don't begin with a dot:
AudioFiles = {}
for file in media_files:
    title = file.split("\\")[-1]
    if len(AudioFiles.keys()):
        audio_id = int(max(AudioFiles.keys()).lstrip('audio')) + 1
    else:
        audio_id = 0

    audio_id = 'audio%i' % audio_id
    AudioFiles[audio_id] = {'title': title}

