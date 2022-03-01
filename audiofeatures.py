import librosa
import soundfile as sf
from speechbrain.pretrained import SepformerSeparation as separator
import speechbrain.pretrained
from speechbrain.pretrained import *

import torchaudio
import os
import werkzeug
import glob
from flask import send_from_directory
from flask_restful import Resource, Api, reqparse
import pathlib

MEDIA_DIR = os.path.join(os.getcwd(), "media")
AUDIO_SEPARATION_SEPFORMER_WHAMR = os.path.join(os.getcwd(), "audio_separation_sepformer_whamr")
AUDIO_SEPARATION_SEPFORMER_WHAM = os.path.join(os.getcwd(), "audio_separation_sepformer_wham")
ENHANCEMENT_SEPFORMER_WSJ0_DIR = os.path.join(os.getcwd(), "enhancement_sepformer_wsj0mix")
ENHANCEMENT_SEPFORMER_WHAMR_DIR = os.path.join(os.getcwd(), "enhancement_sepformer_whamr")
ENHANCEMENT_SEPFORMER_WHAM_DIR = os.path.join(os.getcwd(), "enhancement_sepformer_wham")
ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR = os.path.join(os.getcwd(), "enhancement_metricganplus_voicebank")
SEPARATION_SEPFORMER_WSJ3_DIR = os.path.join(os.getcwd(), "separation_sepformer_wsj03mix")
SEPARATION_SEPFORMER_WSJ2_DIR = os.path.join(os.getcwd(), "separation_sepformer_wsj02mix")
SEPARATION_WHAM_DIR = os.path.join(os.getcwd(), "separation_sepformer_wham")
SEPARATION_WHAMR_DIR = os.path.join(os.getcwd(), "separation_sepformer_whamr")

pathlib.Path(AUDIO_SEPARATION_SEPFORMER_WHAMR).mkdir(parents=True, exist_ok=True)
pathlib.Path(AUDIO_SEPARATION_SEPFORMER_WHAM).mkdir(parents=True, exist_ok=True)
pathlib.Path(ENHANCEMENT_SEPFORMER_WSJ0_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(ENHANCEMENT_SEPFORMER_WHAM_DIR).mkdir(parents=True, exist_ok=True)
pathlib.Path(ENHANCEMENT_SEPFORMER_WHAMR_DIR).mkdir(parents=True, exist_ok=True)
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


def get_filenames(_dir):
    separated_filename_paths = glob.glob(os.path.join(_dir, "*"))
    separated_filenames = []
    for file in separated_filename_paths:
        separated_filenames.append(file.split("\\")[-1])
    return separated_filenames


class AudioseparationSepformerWhamr(Resource):
    """
    ** SepFormer trained on WHAMR!
    This repository provides all the necessary tools to perform audio source separation with a SepFormer model,
    implemented with SpeechBrain, and pretrained on WHAMR! dataset,
    which is basically a version of WSJ0-Mix dataset with environmental noise and reverberation.
    For a better experience we encourage you to learn more about SpeechBrain.
    The model performance is 13.7 dB SI-SNRi on the test set of WHAMR! dataset.
    Release	Test-Set SI-SNRi	Test-Set SDRi
    30-03-21	13.7 dB	12.7 dB

    The system expects input recordings sampled at 8kHz (single channel).
    If your signal has a different sample rate, resample it
    (e.g, using torchaudio or sox) before using the interface.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'sepformer-whamr')
        model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir=model_path)
        est_sources = model.separate_file(path=audiofile_path)
        torchaudio.save(os.path.join(AUDIO_SEPARATION_SEPFORMER_WHAMR, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(AUDIO_SEPARATION_SEPFORMER_WHAMR, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)
        separated_filenames = get_filenames(AUDIO_SEPARATION_SEPFORMER_WHAMR)
        return separated_filenames, 201


class AudioseparationSepformerWhamrDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(AUDIO_SEPARATION_SEPFORMER_WHAMR, filename)
        print(filename_path)
        return send_from_directory(AUDIO_SEPARATION_SEPFORMER_WHAMR, filename, as_attachment=True)


class AudioseparationSepformerWham(Resource):
    """
    ** SepFormer trained on WHAM!
    This repository provides all the necessary tools to perform audio source separation with a SepFormer model,
    implemented with SpeechBrain, and pretrained on WHAM! dataset,
    which is basically a version of WSJ0-Mix dataset with environmental noise.
    For a better experience we encourage you to learn more about SpeechBrain.
    The model performance is 16.3 dB SI-SNRi on the test set of WHAM! dataset.

    Release	Test-Set SI-SNRi	Test-Set SDRi
    09-03-21	16.3 dB	16.7 dB

    The system expects input recordings sampled at 8kHz (single channel).
    If your signal has a different sample rate, resample it (e.g, using torchaudio or sox) before using the interface.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'sepformer-wham')
        model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir=model_path)
        est_sources = model.separate_file(path=audiofile_path)
        torchaudio.save(os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)

        separated_filenames = get_filenames(AUDIO_SEPARATION_SEPFORMER_WHAM)
        return separated_filenames, 201


class AudioseparationSepformerWhamDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(AUDIO_SEPARATION_SEPFORMER_WHAM, filename)
        print(filename_path)
        return send_from_directory(AUDIO_SEPARATION_SEPFORMER_WHAM, filename, as_attachment=True)


class EnhancementSepformerWham(Resource):
    """
    ** SepFormer trained on WHAM! for speech enhancement (8k sampling frequency)
    This repository provides all the necessary tools to perform speech enhancement (denoising) with a SepFormer model,
    implemented with SpeechBrain, and pretrained on WHAM! dataset with 8k sampling frequency,
    which is basically a version of WSJ0-Mix dataset with environmental noise and reverberation in 8k.
    For a better experience we encourage you to learn more about SpeechBrain.
    The given model performance is 14.35 dB SI-SNR on the test set of WHAMR! dataset.
    Release	    |Test-Set SI-SNR	| Test-Set PESQ
    01-12-21	    14.35	             3.07
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'sepformer-wham-enhancement')
        model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir=model_path)
        est_sources = model.separate_file(path=audiofile_path)
        torchaudio.save(os.path.join(ENHANCEMENT_SEPFORMER_WHAM_DIR, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        separated_filenames = get_filenames(ENHANCEMENT_SEPFORMER_WHAM_DIR)
        return separated_filenames, 201


class EnhancementSepformerWhamDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(ENHANCEMENT_SEPFORMER_WHAM_DIR, filename)
        print(filename_path)
        return send_from_directory(ENHANCEMENT_SEPFORMER_WHAM_DIR, filename, as_attachment=True)


class EnhancementSepformerWhamr(Resource):
    """
    ** SepFormer trained on WHAMR! for speech enhancement (8k sampling frequency)
    This repository provides all the necessary tools to perform speech enhancement (denoising + dereverberation)
     with a SepFormer model, implemented with SpeechBrain, and pretrained on WHAMR! dataset with 8k sampling frequency,
     which is basically a version of WSJ0-Mix dataset with environmental noise and reverberation in 8k.
    The given model performance is 10.59 dB SI-SNR on the test set of WHAMR! dataset.
    Release	    |Test-Set SI-SNR	| Test-Set PESQ
    01-12-21	    10.59	                2.84
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'sepformer-whamr-enhancement')
        model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir=model_path)
        est_sources = model.separate_file(path=audiofile_path)
        torchaudio.save(os.path.join(ENHANCEMENT_SEPFORMER_WHAMR_DIR, "EnhancementSepformerWhamr_source1.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        separated_filenames = get_filenames(ENHANCEMENT_SEPFORMER_WHAMR_DIR)
        return separated_filenames, 201


class EnhancementSepformerWhamrDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(ENHANCEMENT_SEPFORMER_WHAMR_DIR, filename)
        print(filename_path)
        return send_from_directory(ENHANCEMENT_SEPFORMER_WHAMR_DIR, filename, as_attachment=True)


class EnhancementMetricganplusVoicebank(Resource):
    """
    ** MetricGAN-trained model for Enhancement
    This repository provides all the necessary tools to perform enhancement with SpeechBrain. For a better experience we encourage you to learn more about SpeechBrain. The model performance is:

    Release	Test PESQ	Test STOI
    21-04-27	3.15	93.0

    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling enhance_file if needed.
    Make sure your input tensor is compliant with the expected sampling rate if you use enhance_batch as in the example.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'metricgan-plus-voicebank')

        model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank", savedir=model_path)

        # Load and add fake batch dimension
        noisy = model.load_audio(audiofile_path).unsqueeze(0)

        # Add relative length tensor
        enhanced = model.enhance_batch(noisy, lengths=torch.tensor([1.]))
        # Saving enhanced signal on disk
        torchaudio.save(os.path.join(ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR, 'enhanced.wav'), enhanced.cpu(), 16000)
        separated_filenames = get_filenames(ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR)
        return separated_filenames, 201


class EnhancementMetricganplusVoicebankDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR, filename)
        print(filename_path)
        return send_from_directory(ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR, filename, as_attachment=True)


class SpeechSeparationSepformerWham(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'sepformer-wham')
        model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir=model_path)
        est_sources = model.separate_file(path=audiofile_path)
        torchaudio.save(os.path.join(SEPARATION_WHAM_DIR, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(SEPARATION_WHAM_DIR, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)
        separated_filenames = get_filenames(SEPARATION_WHAM_DIR)
        return separated_filenames, 201


class SpeechSeparationSepformerWhamDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_WHAM_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_WHAM_DIR, filename, as_attachment=True)


class SpeechSeparationSepformerWhamr(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'sepformer-whamr')
        model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir=model_path)
        est_sources = model.separate_file(path=audiofile_path)
        torchaudio.save(os.path.join(SEPARATION_WHAMR_DIR, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(SEPARATION_WHAMR_DIR, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)
        separated_filenames = get_filenames(SEPARATION_WHAMR_DIR)
        return separated_filenames, 201


class SpeechSeparationSepformerWhamrDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_WHAM_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_WHAM_DIR, filename, as_attachment=True)


class SpeechSeparationSepformerWsj02mix(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'sepformer-wsj02mix')
        model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir=model_path)
        est_sources = model.separate_file(path=audiofile_path)
        torchaudio.save(os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)
        separated_filenames = get_filenames(SEPARATION_SEPFORMER_WSJ2_DIR)
        return separated_filenames, 201


class SpeechSeparationSepformerWsj02mixDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_SEPFORMER_WSJ2_DIR, filename, as_attachment=True)


class SpeechSeparationSepformerWsj03mix(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'sepformer-wsj03mix')
        model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir=model_path)
        est_sources = model.separate_file(path=audiofile_path)
        torchaudio.save(os.path.join(SEPARATION_SEPFORMER_WSJ3_DIR, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(SEPARATION_SEPFORMER_WSJ3_DIR, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)
        torchaudio.save(os.path.join(SEPARATION_SEPFORMER_WSJ3_DIR, "source3hat.wav"), est_sources[:, :, 2].detach().cpu(), 8000)
        separated_filenames = get_filenames(SEPARATION_SEPFORMER_WSJ3_DIR)
        return separated_filenames, 201


class SpeechSeparationSepformerWsj03mixDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_SEPFORMER_WSJ3_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_SEPFORMER_WSJ3_DIR, filename, as_attachment=True)


class VadCrdnnLibriparty(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        y, sr = librosa.load(audiofile_path)
        sr1 = 16000
        y1 = librosa.resample(y, orig_sr=sr, target_sr=sr1)
        filename = 'audio_vad.wav'
        audio_path_16khz = os.path.join(MEDIA_DIR, filename)
        sf.write(audio_path_16khz, y1, sr1)
        model_path = os.path.join('pretrained_models', 'vad-crdnn-libriparty')
        VAD = speechbrain.pretrained.VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir=model_path)
        boundaries = VAD.get_speech_segments(audio_path_16khz)
        # Print the output
        result_filename = 'VAD_file.txt'
        VAD.save_boundaries(boundaries, save_path=result_filename)
        result_file = open(result_filename)
        lines = result_file.readlines()
        return lines, 201


class EmotionRecognitionWav2vec2IEMOCAP(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'emotion-recognition-wav2vec2-IEMOCAP')
        classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                                   pymodule_file="custom_interface.py",
                                   classname="CustomEncoderWav2vec2Classifier",
                                   savedir=model_path)
        out_prob, score, index, text_lab = classifier.classify_file(audiofile_path)
        print(text_lab)
        return text_lab, 201


class AsrWav2vec2CommonvoiceFr(Resource):
    """
    Pipeline description
1) This ASR system is composed of 2 different but linked blocks:
    Tokenizer (unigram) that transforms words into subword units and
    trained with the train transcriptions (train.tsv) of CommonVoice (FR).

2) Acoustic model (wav2vec2.0 + CTC). A pretrained wav2vec 2.0 model
    (LeBenchmark/wav2vec2-FR-7K-large) is combined with two DNN layers and finetuned on CommonVoice FR.
    The obtained final acoustic representation is given to the CTC greedy decoder.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling transcribe_file if needed.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-wav2vec2-commonvoice-fr')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-fr", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile_path)
        return transcribed_file, 201


class AsrWav2vec2CommonvoiceIt(Resource):
    """
    Pipeline description
1) This ASR system is composed of 2 different but linked blocks:
    Tokenizer (unigram) that transforms words into subword units and
    trained with the train transcriptions (train.tsv) of CommonVoice (FR).

2) Acoustic model (wav2vec2.0 + CTC). A pretrained wav2vec 2.0 model
    (LeBenchmark/wav2vec2-FR-7K-large) is combined with two DNN layers and finetuned on CommonVoice It.
    The obtained final acoustic representation is given to the CTC greedy decoder.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling transcribe_file if needed.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-wav2vec2-commonvoice-it')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-it", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile_path)
        return transcribed_file, 201


class AsrWav2vec2CommonvoiceEn(Resource):
    """
   This ASR system is composed of 2 different but linked blocks:
    -- Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions
        (train.tsv) of CommonVoice (EN).
    -- Acoustic model (wav2vec2.0 + CTC/Attention).
    A pretrained wav2vec 2.0 model (wav2vec2-lv60-large) is combined with two DNN layers and finetuned on CommonVoice En.
    The obtained final acoustic representation is given to the CTC and attention decoders.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling transcribe_file if needed.

    -- Pipeline description
    This ASR system is composed of 2 different but linked blocks:
    Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions of LibriSpeech.
    Acoustic model made of a wav2vec2 encoder and a joint decoder with CTC + transformer.
    Hence, the decoding also incorporates the CTC probabilities.
    To Train this system from scratch, see our SpeechBrain recipe.

    The system is trained with recordings sampled at 16kHz (single channel). The code will automatically normalize your
     audio (i.e., resampling + mono channel selection) when calling transcribe_file if needed.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-wav2vec2-commonvoice-en')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 200


class AsrWav2vec2CommonvoiceRw(Resource):
    """
   This ASR system is composed of 2 different but linked blocks:
    -- Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions
        (train.tsv) of CommonVoice (EN).
    -- Acoustic model (wav2vec2.0 + CTC/Attention).
    A pretrained wav2vec 2.0 model (wav2vec2-lv60-large) is combined with two DNN layers and finetuned on CommonVoice En.
    The obtained final acoustic representation is given to the CTC and attention decoders.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling transcribe_file if needed.

    -- Pipeline description
    This ASR system is composed of 2 different but linked blocks:
    Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions of LibriSpeech.
    Acoustic model made of a wav2vec2 encoder and a joint decoder with CTC + transformer.
    Hence, the decoding also incorporates the CTC probabilities.
    To Train this system from scratch, see our SpeechBrain recipe.

    The system is trained with recordings sampled at 16kHz (single channel). The code will automatically normalize your
     audio (i.e., resampling + mono channel selection) when calling transcribe_file if needed.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-wav2vec2-commonvoice-rw')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-rw", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile_path)
        return transcribed_file, 200


class AsrWav2vec2TransformerAishellMandarinChinese(Resource):
    """
    ** Transformer for AISHELL + wav2vec2 (Mandarin Chinese)
    This repository provides all the necessary tools to perform automatic speech recognition from an end-to-end
    system pretrained on AISHELL +wav2vec2 (Mandarin Chinese) within SpeechBrain.
    For a better experience, we encourage you to learn more about SpeechBrain.
    The performance of the model is the following:
    Release	Dev     CER	      Test CER	    GPUs	        Full Results
    05-03-21	   5.19 	    5.58	 2xV100 32GB	    Google Drive(https://drive.google.com/drive/folders/1zlTBib0XEwWeyhaXDXnkqtPsIBI18Uzs?usp=sharing)

    ** Pipeline description
    This ASR system is composed of 2 different but linked blocks:
    1 - Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions of LibriSpeech.
    2 - Acoustic model made of a wav2vec2 encoder and a joint decoder with CTC + transformer.
    Hence, the decoding also incorporates the CTC probabilities.
    To Train this system from scratch, see our SpeechBrain recipe.

    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling transcribe_file if needed.

    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-wav2vec2-transformer-aishell')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-transformer-aishell", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 201


class AsrCrdnntransformerlmLibrispeechEn(Resource):
    """
** CRDNN with CTC/Attention and RNNLM trained on LibriSpeech
This repository provides all the necessary tools to perform automatic speech recognition from an end-to-end system
pretrained on LibriSpeech (EN) within SpeechBrain. For a better experience,
we encourage you to learn more about SpeechBrain.

The performance of the model is the following:

Release	Test clean WER	Test other WER	GPUs
05-03-21	2.90	8.51	1xV100 16GB
** Pipeline description
This ASR system is composed of 3 different but linked blocks:

Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions of LibriSpeech.
Neural language model (Transformer LM) trained on the full 10M words dataset.
Acoustic model (CRDNN + CTC/Attention). The CRDNN architecture is made of N blocks of convolutional neural networks
with normalization and pooling on the frequency domain.
Then, a bidirectional LSTM with projection layers is connected to a final DNN to obtain the final acoustic
representation that is given to the CTC and attention decoders.
The system is trained with recordings sampled at 16kHz (single channel).
The code will automatically normalize your audio
(i.e., resampling + mono channel selection) when calling transcribe_file if needed.


"""

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-crdnn-transformerlm-librispeech')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-crdnn-transformerlm-librispeech", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 201


class AsrCrdnnrnnlmLibrispeechEn(Resource):
    """
    ** CRDNN with CTC/Attention and RNNLM trained on LibriSpeech
    This repository provides all the necessary tools to perform automatic speech recognition from an end-to-end system
    pretrained on LibriSpeech (EN) within SpeechBrain.
    For a better experience we encourage you to learn more about SpeechBrain.
    ---------------------------------------------------------------------------
    The performance of the model is the following:
    Release	Test WER	GPUs
    20-05-22	3.09	1xV100 32GB
    ---------------------------------------------------------------------------
    ** Pipeline description
    This ASR system is composed with 3 different but linked blocks:
    1 - Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions of LibriSpeech.
    Neural language model (RNNLM) trained on the full 10M words dataset.
    2 - Acoustic model (CRDNN + CTC/Attention).
    The CRDNN architecture is made of N blocks of convolutional neural networks with normalisation and pooling on
    the frequency domain.
    Then, a bidirectional LSTM is connected to a final DNN to obtain the final acoustic representation that is given to
    the CTC and attention decoders.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling transcribe_file if needed.
        """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-crdnn-rnnlm-librispeech')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 201


class AsrCrdnnCommonvoiceFr(Resource):
    """
   Pipeline description
This ASR system is composed of 2 different but linked blocks:

-- Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions
    (train.tsv) of CommonVoice (FR).
-- Acoustic model (CRDNN + CTC/Attention).
    The CRDNN architecture is made of N blocks of convolutional neural networks with
    normalization and pooling on the frequency domain.
    Then, a bidirectional LSTM is connected to a final DNN to obtain the final acoustic representation
    that is given to the CTC and attention decoders.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling transcribe_file if needed.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-crdnn-commonvoice-fr')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-fr", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 201


class AsrCrdnnCommonvoiceIt(Resource):
    """
    ** CRDNN with CTC/Attention trained on CommonVoice Italian (No LM)
    This repository provides all the necessary tools to perform automatic speech recognition from an end-to-end system pretrained on CommonVoice (IT) within SpeechBrain. For a better experience, we encourage you to learn more about SpeechBrain.

    The performance of the model is the following:

    Release	Test CER	Test WER	GPUs
    07-03-21	5.40	16.61	2xV100 16GB

    ** Pipeline description
    This ASR system is composed of 2 different but linked blocks:
    1 - Tokenizer (unigram) that transforms words into subword units and
    trained with the train transcriptions (train.tsv) of CommonVoice (IT).
    Acoustic model (CRDNN + CTC/Attention).
    2 - The CRDNN architecture is made of N blocks of convolutional neural networks with normalization
    and pooling on the frequency domain.
    Then, a bidirectional LSTM is connected to a final DNN to obtain the final acoustic representation that is given
    to the CTC and attention decoders.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio (i.e., resampling + mono channel selection)
     when calling transcribe_file if needed.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-crdnn-commonvoice-it')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-it", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 201


class AsrCrdnnCommonvoiceDe(Resource):
    """
    ** CRDNN with CTC/Attention trained on CommonVoice 7.0 German (No LM)
    This repository provides all the necessary tools to perform automatic speech recognition from an end-to-end system pretrained on CommonVoice (German Language) within SpeechBrain. For a better experience, we encourage you to learn more about SpeechBrain. The performance of the model is the following:

    Release	Test CER	Test WER	GPUs
    28.10.21	4.93	15.37	1xV100 16GB

    ** Pipeline description
    This ASR system is composed of 2 different but linked blocks:

    1 - Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions (train.tsv)
    of CommonVoice (DE).
    Acoustic model (CRDNN + CTC/Attention).
    2 - The CRDNN architecture is made of N blocks of convolutional neural networks with normalization
    and pooling on the frequency domain. Then, a bidirectional LSTM is connected to a final DNN to obtain the final
    acoustic representation that is given to the CTC and attention decoders.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio (i.e., resampling + mono channel selection)
    when calling transcribe_file if needed.
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-crdnn-commonvoice-de')
        asr_model = EncoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-de", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 201


class AsrConformerTransformerlmKsponspeechKorean(Resource):
    """
    ** CRDNN with CTC/Attention trained on CommonVoice 7.0 German (No LM)
    This repository provides all the necessary tools to perform automatic speech recognition from an end-to-end system pretrained on CommonVoice (German Language) within SpeechBrain. For a better experience, we encourage you to learn more about SpeechBrain. The performance of the model is the following:

    Release	Test CER	Test WER	GPUs
    28.10.21	4.93	15.37	1xV100 16GB

    ** Pipeline description
    This ASR system is composed of 2 different but linked blocks:

    1 - Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions (train.tsv)
    of CommonVoice (DE).
    Acoustic model (CRDNN + CTC/Attention).
    2 - The CRDNN architecture is made of N blocks of convolutional neural networks with normalization
    and pooling on the frequency domain. Then, a bidirectional LSTM is connected to a final DNN to obtain the final
    acoustic representation that is given to the CTC and attention decoders.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio (i.e., resampling + mono channel selection)
    when calling transcribe_file if needed.
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-conformer-transformerlm-ksponspeech')
        asr_model = EncoderASR.from_hparams(source="ddwkim/asr-conformer-transformerlm-ksponspeech", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 201


class AsrConformerTransformerlmLibrispeechEn(Resource):
    """
    ** Transformer for LibriSpeech (with Transformer LM)
    This repository provides all the necessary tools to perform automatic speech recognition from an end-to-end
    system pretrained on LibriSpeech (EN) within SpeechBrain. For a better experience,
    we encourage you to learn more about SpeechBrain.
    The performance of the model is the following:

    Release	Test clean WER	Test other WER	GPUs
    05-03-21	2.46	5.86	2xV100 32GB

    ** Pipeline description
    This ASR system is composed of 3 different but linked blocks:
    1 - Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions of LibriSpeech.
    Neural language model (Transformer LM) trained on the full 10M words dataset.
    2 - Acoustic model made of a transformer encoder and a joint decoder with CTC + transformer.
    Hence, the decoding also incorporates the CTC probabilities.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling transcribe_file if needed.
    """

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'asr-conformer-transformerlm-ksponspeech')
        asr_model = EncoderASR.from_hparams(source="ddwkim/asr-conformer-transformerlm-ksponspeech", savedir=model_path)
        transcribed_file = asr_model.transcribe_file(audiofile)
        return transcribed_file, 201


class LangidCommonlanguageEcapa(Resource):
    """
    ** Language Identification from Speech Recordings with ECAPA embeddings on CommonLanguage
    This repository provides all the necessary tools to perform language identification from speeech
    recordinfs with SpeechBrain. The system uses a model pretrained on the CommonLanguage dataset (45 languages).
    You can download the dataset here The provided system can recognize the following 45 languages from short speech recordings:
    -Arabic, Basque, Breton, Catalan, Chinese_China, Chinese_Hongkong, Chinese_Taiwan, Chuvash, Czech, Dhivehi, Dutch,
    -English, Esperanto, Estonian, French, Frisian, Georgian, German, Greek, Hakha_Chin, Indonesian,
    -Interlingua, Italian, Japanese, Kabyle, Kinyarwanda, Kyrgyz, Latvian, Maltese, Mangolian, Persian, Polish,
    -Portuguese, Romanian, Romansh_Sursilvan, Russian, Sakha, Slovenian, Spanish, Swedish,
    -Tamil, Tatar, Turkish, Ukranian, Welsh

    ** Pipeline description
    This system is composed of a ECAPA model coupled with statistical pooling.
    A classifier, trained with Categorical Cross-Entropy Loss, is applied on top of that.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling classify_file if needed.
    Make sure your input tensor is compliant with the expected sampling rate if you use encode_batch and classify_batch.

    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'lang-id-commonlanguage_ecapa')
        classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir=model_path)
        out_prob, score, index, text_lab = classifier.classify_file(audiofile_path)
        print(text_lab)
        return text_lab, 201


class LangidVoxLingua107Ecapa(Resource):
    """
    ** VoxLingua107 ECAPA-TDNN Spoken Language Identification Model
    - Model description
    This is a spoken language recognition model trained on the VoxLingua107 dataset using SpeechBrain.
    The model uses the ECAPA-TDNN architecture that has previously been used for speaker recognition.
    However, it uses more fully connected hidden layers after the embedding layer,
    and cross-entropy loss was used for training.
    We observed that this improved the performance of extracted utterance embeddings for downstream tasks.
    The system is trained with recordings sampled at 16kHz (single channel).
    The code will automatically normalize your audio
    (i.e., resampling + mono channel selection) when calling classify_file if needed.
    The model can classify a speech utterance according to the language spoken.
    It covers 107 different languages
    ( Abkhazian, Afrikaans, Amharic, Arabic, Assamese, Azerbaijani, Bashkir, Belarusian, Bulgarian, Bengali, Tibetan,
    Breton, Bosnian, Catalan, Cebuano, Czech, Welsh, Danish, German, Greek, English, Esperanto, Spanish, Estonian, Basque,
    Persian, Finnish, Faroese, French, Galician, Guarani, Gujarati, Manx, Hausa, Hawaiian, Hindi, Croatian, Haitian,
    Hungarian, Armenian, Interlingua, Indonesian, Icelandic, Italian, Hebrew, Japanese, Javanese, Georgian, Kazakh,
    Central Khmer, Kannada, Korean, Latin, Luxembourgish, Lingala, Lao, Lithuanian, Latvian, Malagasy, Maori, Macedonian,
    Malayalam, Mongolian, Marathi, Malay, Maltese, Burmese, Nepali, Dutch, Norwegian Nynorsk, Norwegian, Occitan, Panjabi,
    Polish, Pushto, Portuguese, Romanian, Russian, Sanskrit, Scots, Sindhi, Sinhala, Slovak, Slovenian, Shona, Somali,
    Albanian, Serbian, Sundanese, Swedish, Swahili, Tamil, Telugu, Tajik, Thai, Turkmen, Tagalog, Turkish, Tatar, Ukrainian,
    Urdu, Uzbek, Vietnamese, Waray, Yiddish, Yoruba, Mandarin Chinese).
    """
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        model_path = os.path.join('pretrained_models', 'lang-id-voxlingua107-ecapa')
        language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=model_path)
        # Download Thai language sample from Omniglot and cvert to suitable form
        signal = language_id.load_audio(audiofile_path)
        prediction = language_id.classify_batch(signal)
        lang = prediction[3]
        return lang, 201
