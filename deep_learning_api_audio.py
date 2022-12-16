import json
from pprint import pprint

import werkzeug
from flask import send_from_directory
from flask_restful import Resource, reqparse
from deep_learning_features_audio import *
from deep_learning_dict_api import AudioAnalysisAPI


def get_filenames(_dir):
    separated_filename_paths = glob.glob(os.path.join(_dir, "*"))
    separated_filenames = []
    for file in separated_filename_paths:
        separated_filenames.append(file.split("\\")[-1])
    return separated_filenames


class ApiDeepLearningFeaturesList(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self):
        api_list = AudioAnalysisAPI.copy()
        for dict_element in api_list.values():
            try:
                del dict_element['function']
            except Exception as e:
                print(e)
        dl_api_list = json.dumps(api_list)
        return dl_api_list


class ApiAudioseparationSepformerWhamr(Resource):
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
        separated_file_paths = audioseparation_sepformer_whamr(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]
        print(separated_filenames)
        return separated_filenames, 201


class ApiAudioseparationSepformerWhamrDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(AUDIO_SEPARATION_SEPFORMER_WHAMR, filename)
        print(filename_path)
        return send_from_directory(AUDIO_SEPARATION_SEPFORMER_WHAMR, filename, as_attachment=True)


class ApiAudioseparationSepformerWham(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = audioseparation_sepformer_wham(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]
        print(separated_filenames)
        return separated_filenames, 201


class ApiAudioseparationSepformerWhamDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(AUDIO_SEPARATION_SEPFORMER_WHAM, filename)
        print(filename_path)
        return send_from_directory(AUDIO_SEPARATION_SEPFORMER_WHAM, filename, as_attachment=True)


class ApiEnhancementSepformerWham(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = enhancement_sepformer_wham(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]
        return separated_filenames, 201


class ApiEnhancementSepformerWhamDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(ENHANCEMENT_SEPFORMER_WHAM_DIR, filename)
        print(filename_path)
        return send_from_directory(ENHANCEMENT_SEPFORMER_WHAM_DIR, filename, as_attachment=True)


class ApiEnhancementSepformerWhamr(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = enhancement_sepformer_whamr(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]

        return separated_filenames, 201


class ApiEnhancementSepformerWhamrDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(ENHANCEMENT_SEPFORMER_WHAMR_DIR, filename)
        print(filename_path)
        return send_from_directory(ENHANCEMENT_SEPFORMER_WHAMR_DIR, filename, as_attachment=True)


class ApiEnhancementSepformerWhamr16k(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = enhancement_sepformer_whamr_16k(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]

        return separated_filenames, 201


class ApiEnhancementSepformerWhamrDownload16k(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(ENHANCEMENT_SEPFORMER_WHAMR_16k_DIR, filename)
        print(filename_path)
        return send_from_directory(ENHANCEMENT_SEPFORMER_WHAMR_16k_DIR, filename, as_attachment=True)


class ApiEnhancementMetricganplusVoicebank(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = enhancement_metricganplus_voicebank(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]

        return separated_filenames, 201


class ApiEnhancementMetricganplusVoicebankDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR, filename)
        print(filename_path)
        return send_from_directory(ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR, filename, as_attachment=True)


class ApiSpeechSeparationSepformerWham(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = speechseparation_sepformer_wham(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]

        return separated_filenames, 201


class ApiSpeechSeparationSepformerWhamDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_WHAM_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_WHAM_DIR, filename, as_attachment=True)


class ApiSpeechSeparationSepformerWhamr(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = speechseparation_sepformer_whamr(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]

        return separated_filenames, 201


class ApiSpeechSeparationSepformerWhamrDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_WHAMR_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_WHAMR_DIR, filename, as_attachment=True)


class ApiSpeechSeparationSepformerWsj02mix(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = speechseparation_sepformer_wsj02mix(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]
        return separated_filenames, 201


class ApiSpeechSeparationSepformerWsj02mixDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_SEPFORMER_WSJ2_DIR, filename, as_attachment=True)


class ApiSpeechSeparationSepformerWsj03mix(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        separated_file_paths = speechseparation_sepformer_wsj03mix(audiofile_path)
        separated_filenames = [os.path.split(x)[-1] for x in separated_file_paths]

        return separated_filenames, 201


class ApiSpeechSeparationSepformerWsj03mixDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(SEPARATION_SEPFORMER_WSJ3_DIR, filename)
        print(filename_path)
        return send_from_directory(SEPARATION_SEPFORMER_WSJ3_DIR, filename, as_attachment=True)


class ApiVadCrdnnLibripartyCleaned(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        vad_clean_file = vad_crdnn_libriparty_cleaned(audiofile_path)
        vad_clean_file = [os.path.split(x)[-1] for x in vad_clean_file]
        return vad_clean_file, 201


class ApiVadCrdnnLibripartyCleanedDownload(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def get(self, filename):
        print("==> filename:{}".format(filename))
        filename_path = os.path.join(VAD_CRDNN, filename)
        print(filename_path)
        return send_from_directory(VAD_CRDNN, filename, as_attachment=True)


class ApiVadCrdnnLibriparty(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        lines = vad_crdnn_libriparty(audiofile_path)
        return lines, 201


class ApiEmotionRecognitionWav2vec2IEMOCAP(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        text_lab = emotion_recognition__wav2vec2__iemocap(audiofile_path)
        return text_lab, 201


class ApiAsrWav2vec2CommonvoiceFr(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__commonvoice_fr(audiofile_path)
        return transcribed_file, 201


class ApiAsrWav2vec2CommonvoiceIt(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__commonvoice_it(audiofile_path)
        return transcribed_file, 201


class ApiAsrWav2vec2CommonvoiceEn(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__commonvoice_en(audiofile_path)
        return transcribed_file, 200


class ApiAsrWav2vec2CommonvoiceRw(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__commonvoice_rw(audiofile_path)
        return transcribed_file, 200


class ApiAsrWav2vec2VoxpopuliDe(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__voxpopuli_de(audiofile_path)
        return transcribed_file, 200


class ApiAsrWav2vec2VoxpopuliEn(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__voxpopuli_en(audiofile_path)
        return transcribed_file, 200


class ApiAsrWav2vec2VoxpopuliEs(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__voxpopuli_es(audiofile_path)
        return transcribed_file, 200


class ApiAsrWav2vec2VoxpopuliFr(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__voxpopuli_fr(audiofile_path)
        return transcribed_file, 201


class ApiAsrWav2vec2VoxpopuliIt(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__wav2vec2__voxpopuli_it(audiofile_path)
        return transcribed_file, 201


class ApiAsrWav2vec2TransformerAishellMandarinChinese(Resource):
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
        transcribed_file = asr__wav2vec2_transformer__aishell_mandarin_chinese(audiofile_path)
        return transcribed_file, 201


class ApiAsrCrdnntransformerlmLibrispeechEn(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__crdnn_transformerlm__librispeech_en(audiofile_path)
        return transcribed_file, 201


class ApiAsrCrdnnrnnlmLibrispeechEn(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__crdnn_rnn_lm__librispeech_en(audiofile_path)
        return transcribed_file, 201


class ApiAsrCrdnnCommonvoiceFr(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__crdnn__commonvoice_fr(audiofile_path)
        return transcribed_file, 201


class ApiAsrCrdnnCommonvoiceIt(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__crdnn__commonvoice_it(audiofile_path)
        return transcribed_file, 201


class ApiAsrCrdnnCommonvoiceDe(Resource):
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
        transcribed_file = asr__crdnn__commonvoice_de(audiofile_path)
        return transcribed_file, 201


class ApiAsrConformerTransformerlmKsponspeechKorean(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__conformer_transformer_lm__ksponspeech_korean(audiofile_path)
        return transcribed_file, 201


class ApiAsrConformerTransformerlmLibrispeechEn(Resource):
    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed_file = asr__conformer_transformer_lm__librispeech_en(audiofile_path)
        return transcribed_file, 201


class ApiLangidEcapaCommonlanguage(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        text_lab = language_identification__ecapa__commonlanguage(audiofile_path)
        return text_lab, 201


class ApiLangidEcapaVoxLingua107(Resource):
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
        lang = language_identification__ecapa__vox_lingua107(audiofile_path)
        return lang, 201


class ApiLangidToAsr(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        transcribed = lang_id__to__asr(audiofile_path)
        return transcribed, 201



class ApiSpeakerVerificationData2VecAudioForXVectorLibrispeechEn(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('threshold')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile= args.get("audiofile")
        audiofile_path= os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        threshold = args.get("threshold")
        if len(os.listdir(SPEAKER_VERIFICATION_DATASET)) == 0 :
            return "No data in speaker verification dataset",201
        else:
            for audiofile in os.listdir(SPEAKER_VERIFICATION_DATASET):
                audiofile_path2 = os.path.join(SPEAKER_VERIFICATION_DATASET, audiofile)
                result = result + "audiofile=" + audiofile + " :" + speaker_verification__Wav2Vec2ForXVector__wav2vec2_base_superb_sv(audiofile_path, audiofile_path2, threshold)+"\n"
        return result, 201


class ApiSpeakerVerificationWav2Vec2ForXVectorWav2Vec2BaseSuperbSv(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        result = ""
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('threshold')
        self.parser.add_argument('title')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        threshold = args.get("threshold")
        if len(os.listdir(SPEAKER_VERIFICATION_DATASET)) == 0 :
            return "No data in speaker verification dataset",201
        else:
            for audiofile in os.listdir(SPEAKER_VERIFICATION_DATASET):
                audiofile_path2 = os.path.join(SPEAKER_VERIFICATION_DATASET, audiofile)
                result = result + "audiofile=" + audiofile + " :" + speaker_verification__Wav2Vec2ForXVector__wav2vec2_base_superb_sv(audiofile_path, audiofile_path2, threshold)+"\n"
        return result, 201


class ApiSpeakerVerificationWav2Vec2ConformerForXVector(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()

    def post(self):
        self.parser.add_argument("audiofile", type=werkzeug.datastructures.FileStorage, location='files')
        self.parser.add_argument('threshold')
        self.parser.add_argument('title1')
        args = self.parser.parse_args()
        audiofile = args.get("audiofile")
        audiofile_path = os.path.join(MEDIA_DIR, audiofile.filename)
        audiofile.save(audiofile_path)
        threshold = args.get("threshold")
        if len(os.listdir(SPEAKER_VERIFICATION_DATASET)) == 0 :
            return "No data in speaker verification dataset",201
        else:
            for audiofile in os.listdir(SPEAKER_VERIFICATION_DATASET):
                audiofile_path2 = os.path.join(SPEAKER_VERIFICATION_DATASET, audiofile)
                result = result + "audiofile=" + audiofile + " :" + speaker_verification__Wav2Vec2ForXVector__wav2vec2_base_superb_sv(audiofile_path, audiofile_path2, threshold)+"\n"
        return result, 201
