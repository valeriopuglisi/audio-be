from flask import send_from_directory
from flask_restful import Resource, Api, reqparse
import werkzeug
import glob
import librosa
from librosa import display
import matplotlib.pyplot as plt
import numpy as np
import os


PREPROCESS_DIR = os.path.join(os.getcwd(), "preprocess")


class LinearFrequencyPowerSpectrogram(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y,)), ref=np.max)
        img = display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax[1])
        ax[1].set(title='Linear-frequency power spectrogram')
        ax[1].label_outer()
        fig.tight_layout()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        filename = 'LinearFrequencyPowerSpectrogram.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class LogFrequencyPowerSpectrogram(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        hop_length = 1024
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y, hop_length=hop_length)), ref=np.max)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        img = display.specshow(D, y_axis='log', sr=sr, hop_length=hop_length, x_axis='time', ax=ax[1])
        ax[1].set(title='Log-frequency power spectrogram')
        ax[1].label_outer()
        fig.tight_layout()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        filename = 'LogFrequencyPowerSpectrogram.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class ChromaStft(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        S = np.abs(librosa.stft(y, n_fft=4096))
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        img = display.specshow(chroma, y_axis='chroma', sr=sr, x_axis='time', ax=ax[1])
        ax[1].set(title='ChromaSTFT')
        ax[1].label_outer()
        fig.tight_layout()
        fig.colorbar(img, ax=ax)
        filename = 'ChromaSTFT.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class ChromaCQT(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        fig, ax = plt.subplots(nrows=3, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        display.specshow(chroma_stft, y_axis='chroma', sr=sr, x_axis='time', ax=ax[1])
        ax[1].set(title='Chroma STFT')
        ax[1].label_outer()
        img = librosa.display.specshow(chroma_cqt, y_axis='chroma',  sr=sr, x_axis='time', ax=ax[2])
        ax[2].set(title='Chroma CQT')
        ax[2].label_outer()
        fig.tight_layout()
        fig.colorbar(img, ax=ax)
        filename = 'ChromaCQT.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class ChromaCENS(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12, n_fft=4096)
        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        display.specshow(chroma_stft, y_axis='chroma', sr=sr, x_axis='time', ax=ax[1], cmap="coolwarm")
        ax[1].set(title='Chroma STFT')
        ax[1].label_outer()
        img = librosa.display.specshow(chroma_cqt, y_axis='chroma',  sr=sr, x_axis='time', ax=ax[2], cmap="coolwarm")
        ax[2].set(title='Chroma CQT')
        ax[2].label_outer()
        img = librosa.display.specshow(chroma_cens, y_axis='chroma', sr=sr, x_axis='time', ax=ax[3], cmap="coolwarm")
        ax[3].set(title='Chroma CENS')
        ax[3].label_outer()
        fig.tight_layout()
        # fig.colorbar(img, ax=ax)
        filename = 'ChromaCENS.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class Melspectrogram(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        display.specshow(melspectrogram, y_axis='mel', sr=sr, x_axis='time', ax=ax[1])
        ax[1].set(title='Melspectrogram')
        ax[1].label_outer()
        fig.tight_layout()
        # fig.colorbar(img, ax=ax)
        filename = 'Melspectrogram.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)
