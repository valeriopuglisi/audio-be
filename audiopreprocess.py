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

        D = np.abs(librosa.stft(y)) ** 2
        S = librosa.feature.melspectrogram(S=D, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        fig, ax = plt.subplots(nrows=3, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        display.specshow(melspectrogram, y_axis='mel', sr=sr, x_axis='time', ax=ax[1])
        ax[1].set(title='Melspectrogram')
        ax[1].label_outer()
        img = librosa.display.specshow(S_dB, x_axis='time',  y_axis='mel', sr=sr, fmax=8000, ax=ax[2])
        ax[2].set(title='Mel-frequency spectrogram')
        ax[2].label_outer()

        fig.tight_layout()
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        # fig.colorbar(img, ax=ax)
        filename = 'Melspectrogram.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class MelFrequencySpectrogram(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        D = np.abs(librosa.stft(y)) ** 2
        S = librosa.feature.melspectrogram(S=D, sr=sr)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        img = librosa.display.specshow(S_dB, x_axis='time',  y_axis='mel', sr=sr, fmax=8000, ax=ax[1])
        ax[1].set(title='Mel-frequency spectrogram')
        ax[1].label_outer()
        fig.tight_layout()
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        # fig.colorbar(img, ax=ax)
        filename = 'Mel-frequency spectrogram.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class MFCC(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        librosa.feature.mfcc(S=librosa.power_to_db(S))
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        display.waveshow(y, sr=sr, ax=ax[0])
        img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, fmax=8000, ax=ax[1])
        ax[1].set(title='MFCC')
        ax[1].label_outer()
        fig.tight_layout()
        fig.colorbar(img, ax=[ax[1]])
        filename = 'MFCC.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class CompareDCTBases(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        m_slaney = librosa.feature.mfcc(y=y, sr=sr, dct_type=2)
        m_htk = librosa.feature.mfcc(y=y, sr=sr, dct_type=3)
        fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
        img1 = librosa.display.specshow(m_slaney, x_axis='time', ax=ax[0])
        ax[0].set(title='RASTAMAT / Auditory toolbox (dct_type=2)')

        img2 = librosa.display.specshow(m_htk, x_axis='time', ax=ax[1])
        ax[1].set(title='HTK-style (dct_type=3)')
        fig.tight_layout()
        fig.colorbar(img1, ax=[ax[0]])
        fig.colorbar(img2, ax=[ax[1]])

        filename = 'CompareDCTBases.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class RootMeanSquare(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        librosa.feature.rms(y=y)
        S, phase = librosa.magphase(librosa.stft(y))
        rms = librosa.feature.rms(S=S)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        times = librosa.times_like(rms)
        ax[0].semilogy(times, rms[0], label='RMS Energy')
        ax[0].set(xticks=[])
        ax[0].set(title='Root Mean Square Energy')
        ax[0].legend()
        ax[0].label_outer()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set(title='log Power spectrogram')
        filename = 'RootMeanSquare.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class SpectralCentroid(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)

        # From time - series input:
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        # From spectrogram input:
        S, phase = librosa.magphase(librosa.stft(y=y))
        # cent = librosa.feature.spectral_centroid(S=S)

        # Using variable bin center frequencies:
        # freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
        # cent = librosa.feature.spectral_centroid(S=np.abs(D), freq=freqs)

        times = librosa.times_like(cent)
        fig, ax = plt.subplots()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax)
        ax.plot(times, cent.T, label='Spectral centroid', color='w')
        ax.legend(loc='upper right')
        ax.set(title='log Power spectrogram')

        filename = 'SpectralCentroid.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class SpectralBandwidth(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)

        # From time - series input
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        # From spectrogram input
        S, phase = librosa.magphase(librosa.stft(y=y))
        librosa.feature.spectral_bandwidth(S=S)

        # Using variable bin center frequencies
        freqs, times, D = librosa.reassigned_spectrogram(y, fill_nan=True)
        librosa.feature.spectral_bandwidth(S=np.abs(D), freq=freqs)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        times = librosa.times_like(spec_bw)
        centroid = librosa.feature.spectral_centroid(S=S)
        ax[0].semilogy(times, spec_bw[0], label='Spectral bandwidth')
        ax[0].set(ylabel='Hz', xticks=[], xlim=[times.min(), times.max()])
        ax[0].legend()
        ax[0].label_outer()
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[1])
        ax[1].set(title='log Power spectrogram')
        ax[1].fill_between(times, centroid[0] - spec_bw[0], centroid[0] + spec_bw[0], alpha=0.5, label='Centroid +- bandwidth')
        ax[1].plot(times, centroid[0], label='Spectral centroid', color='w')
        ax[1].legend(loc='lower right')

        filename = 'SpectralBandwidth.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)


class SpectralContrast(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        y, sr = librosa.load(audiofile)
        S = np.abs(librosa.stft(y))
        contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
        fig, ax = plt.subplots(nrows=2, sharex=True)
        img1 = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='log', x_axis='time', ax=ax[0])
        fig.colorbar(img1, ax=[ax[0]], format='%+2.0f dB')
        ax[0].set(title='Power spectrogram')
        ax[0].label_outer()
        img2 = librosa.display.specshow(contrast, x_axis='time', ax=ax[1])
        fig.colorbar(img2, ax=[ax[1]])
        ax[1].set(ylabel='Frequency bands', title='Spectral contrast')
        filename = 'SpectralContrast.png'
        fig.savefig(os.path.join(PREPROCESS_DIR, filename))
        return send_from_directory(PREPROCESS_DIR, filename, as_attachment=True)
