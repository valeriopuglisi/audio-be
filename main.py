from flask import Flask, send_file, send_from_directory
from flask_restful import Resource, Api, reqparse
import os
import werkzeug
import glob
import librosa
from librosa import display
import matplotlib.pyplot as plt
from audiofiles import AudioFilesList, AudioFileDownload
import numpy as np

from audiopreprocess import LinearFrequencyPowerSpectrogram, LogFrequencyPowerSpectrogram, ChromaStft, ChromaCQT, \
    ChromaCENS, Melspectrogram, MelFrequencySpectrogram, MFCC, CompareDCTBases, RootMeanSquare, SpectralCentroid,\
    SpectralBandwidth, SpectralContrast

app = Flask(__name__)
api = Api(app)

MEDIA_DIR = os.path.join(os.getcwd(), "media")
SEPARATION_DIR = os.path.join(os.getcwd(), "separation")
# print(MEDIA_DIR)
# print(SEPARATION_DIR)
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


def get_separated_filenames():
    separated_filename_paths = glob.glob(os.path.join(os.getcwd(), "separation", "*"))
    separated_filenames = []
    for file in separated_filename_paths:
        separated_filenames.append(file.split("\\")[-1])
    return separated_filenames


class AudioSeparation(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile.save(os.path.join(SEPARATION_DIR, audiofile.filename))
        separated_filenames = get_separated_filenames()
        return separated_filenames, 201


class AudioSeparationDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_DIR, filename, as_attachment=True)


api.add_resource(AudioFilesList, '/api/audiofiles')
api.add_resource(AudioFileDownload, '/api/audiofiles/<filename>')

api.add_resource(AudioSeparation, '/api/audioseparation')
api.add_resource(AudioSeparationDownload, '/api/audioseparation/<filename>')

api.add_resource(LinearFrequencyPowerSpectrogram, '/api/preprocess/linear_frequency_power_spectrogram')
api.add_resource(LogFrequencyPowerSpectrogram, '/api/preprocess/log_frequency_power_spectrogram')
api.add_resource(ChromaStft, '/api/preprocess/chroma_stft')
api.add_resource(ChromaCQT, '/api/preprocess/chroma_cqt')
api.add_resource(ChromaCENS, '/api/preprocess/chroma_cens')
api.add_resource(Melspectrogram, '/api/preprocess/melspectrogram')
api.add_resource(MelFrequencySpectrogram, '/api/preprocess/melfrequencyspectrogram')
api.add_resource(MFCC, '/api/preprocess/mfcc')
api.add_resource(CompareDCTBases, '/api/preprocess/comparedct')
api.add_resource(RootMeanSquare, '/api/preprocess/rms')
api.add_resource(SpectralCentroid, '/api/preprocess/spectral_centroid')
api.add_resource(SpectralBandwidth, '/api/preprocess/spectral_bandwidth')
api.add_resource(SpectralContrast, '/api/preprocess/spectral_contrast')


if __name__ == '__main__':
    app.run(debug=True)