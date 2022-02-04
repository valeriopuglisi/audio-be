from flask import Flask, send_file, send_from_directory
from flask_restful import Resource, Api, reqparse
import os
import werkzeug
import glob
import librosa
from librosa import display
import matplotlib.pyplot as plt

from audiofeatures import SpeechSeparationSepformerWsj03mix, AudioSeparationDownload
from audiofiles import AudioFilesList, AudioFileDownload
import numpy as np

from audiopreprocess import LinearFrequencyPowerSpectrogram, LogFrequencyPowerSpectrogram, ChromaStft, ChromaCQT, \
    ChromaCENS, Melspectrogram, MelFrequencySpectrogram, MFCC, CompareDCTBases, RootMeanSquare, SpectralCentroid,\
    SpectralBandwidth, SpectralContrast

app = Flask(__name__)
api = Api(app)


api.add_resource(AudioFilesList, '/api/audiofiles')
api.add_resource(AudioFileDownload, '/api/audiofiles/<filename>')

api.add_resource(SpeechSeparationSepformerWsj03mix, '/api/audioseparation/sepformer_wsj03mix')
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