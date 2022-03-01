from flask import Flask, send_file, send_from_directory
from flask_restful import Resource, Api, reqparse
import os
import werkzeug
import glob
import librosa
from librosa import display
import matplotlib.pyplot as plt

from audiofeatures import *

from audiofiles import *
import numpy as np
from audiopreprocess import *

app = Flask(__name__)
api = Api(app)


api.add_resource(AudioFilesList, '/api/audiofiles')
api.add_resource(AudioFileDownload, '/api/audiofiles/<filename>')


# - DEEP LEARNING SPEECH ENHANCEMENT -----------------------------------------------------------------------------------
api.add_resource(EnhancementSepformerWham,
                 '/api/speech_enhancement/enhancement_sepformer_wham')
api.add_resource(EnhancementSepformerWhamDownload,
                 '/api/speech_enhancement/enhancement_sepformer_wham/<filename>')
api.add_resource(EnhancementSepformerWhamr,
                 '/api/speech_enhancement/enhancement_sepformer_whamr')
api.add_resource(EnhancementSepformerWhamrDownload,
                 '/api/speech_enhancement/enhancement_sepformer_whamr/<filename>')
api.add_resource(EnhancementMetricganplusVoicebank,
                 '/api/speech_enhancement/enhancement_metricganplus_voicebank')
api.add_resource(EnhancementMetricganplusVoicebankDownload,
                 '/api/speech_enhancement/enhancement_metricganplus_voicebank/<filename>')
# ----------------------------------------------------------------------------------------------------------------------


# - DEEP LEARNING SPEECH SEPARATION ------------------------------------------------------------------------------------
api.add_resource(SpeechSeparationSepformerWham,
                 '/api/audioseparation/speech_separation_sepformer_wham')
api.add_resource(SpeechSeparationSepformerWhamDownload,
                 '/api/audioseparation/speech_separation_sepformer_wham/<filename>')
api.add_resource(SpeechSeparationSepformerWhamr,
                 '/api/audioseparation/speech_separation_sepformer_whamr')
api.add_resource(SpeechSeparationSepformerWhamrDownload,
                 '/api/audioseparation/speech_separation_sepformer_whamr/<filename>')
api.add_resource(SpeechSeparationSepformerWsj02mix,
                 '/api/audioseparation/speech_separation_sepformer_wsj02mix')
api.add_resource(SpeechSeparationSepformerWsj02mixDownload,
                 '/api/audioseparation/speech_separation_sepformer_wsj02mix/<filename>')
api.add_resource(SpeechSeparationSepformerWsj03mix,
                 '/api/audioseparation/speech_separation_sepformer_wsj03mix')
api.add_resource(SpeechSeparationSepformerWsj03mixDownload,
                 '/api/audioseparation/speech_separation_sepformer_wsj03mix/<filename>')
# ----------------------------------------------------------------------------------------------------------------------

# - DEEP LEARNING AUTOMATIC SPEECH RECOGNITION -------------------------------------------------------------------------
api.add_resource(AsrCrdnnCommonvoiceIt,
                 '/api/automatic_speech_recognition/asr_crdnn_commonvoice_it')
api.add_resource(AsrCrdnnCommonvoiceFr,
                 '/api/automatic_speech_recognition/asr_crdnn_commonvoice_fr')
api.add_resource(AsrCrdnnCommonvoiceDe,
                 '/api/automatic_speech_recognition/asr_crdnn_commonvoice_de')
api.add_resource(AsrWav2vec2CommonvoiceIt,
                 '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_it')
api.add_resource(AsrWav2vec2CommonvoiceFr,
                 '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_fr')
api.add_resource(AsrWav2vec2CommonvoiceEn,
                 '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_en')
api.add_resource(AsrWav2vec2CommonvoiceRw,
                 '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_rw')
api.add_resource(AsrWav2vec2TransformerAishellMandarinChinese,
                 '/api/automatic_speech_recognition/asr_wav2vec2_transformer_aishell_mandarin_chinese')
api.add_resource(AsrConformerTransformerlmLibrispeechEn,
                 '/api/automatic_speech_recognition/asr_conformer_transformerlm_librispeech_en')
api.add_resource(AsrCrdnntransformerlmLibrispeechEn,
                 '/api/automatic_speech_recognition/asr_crdnntransformerlm_librispeech_en')
api.add_resource(AsrCrdnnrnnlmLibrispeechEn, '/api/automatic_speech_recognition/asr_crdnnrnnlm_librispeech_en' )
# ----------------------------------------------------------------------------------------------------------------------


# - DEEP LEARNING LANGUAGE IDENTIFICATION -------------------------------------------------------------------------
api.add_resource(LangidCommonlanguageEcapa,
                 '/api/language_id/langid_commonlanguage_ecapa')
api.add_resource(LangidVoxLingua107Ecapa,
                 '/api/language_id/langid_voxlingua107_ecapa')
# ----------------------------------------------------------------------------------------------------------------------

# - DEEP LEARNING VOICE ACTIVITY DETECTION -------------------------------------------------------------------------
api.add_resource(VadCrdnnLibriparty,
                 '/api/voice_activity_detection/vad_crdnn_libriparty')
# ----------------------------------------------------------------------------------------------------------------------

api.add_resource(EmotionRecognitionWav2vec2IEMOCAP,
                 '/api/emotion_recognition/wav2vec2_IEMOCAP')
# ----------------------------------------------------------------------------------------------------------------------

# - LIBROSA AUDIO FEATURES EXTRACTION ----------------------------------------------------------------------------------
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
# ----------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    app.run(debug=True)