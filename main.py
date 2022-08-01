from deep_learning_api_audio import *
from audiofiles import *
from deep_learning_api_dataset import *
from deep_learning_api_metrics import *
from deep_learning_api_preprocess_audio import *
from deep_learning_api_pipelines import *
from deep_learning_api_report import Reports, Report

app = Flask(__name__)
api = Api(app)

api.add_resource(ApiDeepLearningFeaturesList, '/api/deep-learning-features')
api.add_resource(ApiDatasetsList, '/api/datasets')
api.add_resource(ApiMetricsList, '/api/metrics')
api.add_resource(ApiEvaluationsList, '/api/evaluations')
api.add_resource(AudioFilesList, '/api/audiofiles')
api.add_resource(AudioFileDownload, '/api/audiofiles/<filename>')
api.add_resource(SavePipeline, '/api/utils/save-pipeline')
api.add_resource(Pipelines, '/api/stored-pipelines')
api.add_resource(Pipeline, '/api/stored-pipelines/<id>')
api.add_resource(Reports, '/api/reports')
api.add_resource(Report, '/api/report/<report_id>')


# - DEEP LEARNING SPEECH ENHANCEMENT -----------------------------------------------------------------------------------
api.add_resource(ApiEnhancementSepformerWham,
                 '/api/speech_enhancement/enhancement_sepformer_wham')
api.add_resource(ApiEnhancementSepformerWhamDownload,
                 '/api/speech_enhancement/enhancement_sepformer_wham/<filename>')
api.add_resource(ApiEnhancementSepformerWhamr,
                 '/api/speech_enhancement/enhancement_sepformer_whamr')
api.add_resource(ApiEnhancementSepformerWhamrDownload,
                 '/api/speech_enhancement/enhancement_sepformer_whamr/<filename>')
api.add_resource(ApiEnhancementSepformerWhamr16k,
                 '/api/speech_enhancement/enhancement_sepformer_whamr_16k')
api.add_resource(ApiEnhancementSepformerWhamrDownload16k,
                 '/api/speech_enhancement/enhancement_sepformer_whamr_16k/<filename>')

api.add_resource(ApiEnhancementMetricganplusVoicebank,
                 '/api/speech_enhancement/enhancement_metricganplus_voicebank')
api.add_resource(ApiEnhancementMetricganplusVoicebankDownload,
                 '/api/speech_enhancement/enhancement_metricganplus_voicebank/<filename>')
# ----------------------------------------------------------------------------------------------------------------------


# - DEEP LEARNING SPEECH SEPARATION ------------------------------------------------------------------------------------
api.add_resource(ApiSpeechSeparationSepformerWham,
                 '/api/audioseparation/speech_separation_sepformer_wham')
api.add_resource(ApiSpeechSeparationSepformerWhamDownload,
                 '/api/audioseparation/speech_separation_sepformer_wham/<filename>')
api.add_resource(ApiSpeechSeparationSepformerWhamr,
                 '/api/audioseparation/speech_separation_sepformer_whamr')
api.add_resource(ApiSpeechSeparationSepformerWhamrDownload,
                 '/api/audioseparation/speech_separation_sepformer_whamr/<filename>')
api.add_resource(ApiSpeechSeparationSepformerWsj02mix,
                 '/api/audioseparation/speech_separation_sepformer_wsj02mix')
api.add_resource(ApiSpeechSeparationSepformerWsj02mixDownload,
                 '/api/audioseparation/speech_separation_sepformer_wsj02mix/<filename>')
api.add_resource(ApiSpeechSeparationSepformerWsj03mix,
                 '/api/audioseparation/speech_separation_sepformer_wsj03mix')
api.add_resource(ApiSpeechSeparationSepformerWsj03mixDownload,
                 '/api/audioseparation/speech_separation_sepformer_wsj03mix/<filename>')
# ----------------------------------------------------------------------------------------------------------------------

# - DEEP LEARNING AUTOMATIC SPEECH RECOGNITION -------------------------------------------------------------------------
api.add_resource(ApiAsrCrdnnCommonvoiceIt,
                 '/api/automatic_speech_recognition/asr_crdnn_commonvoice_it')
api.add_resource(ApiAsrCrdnnCommonvoiceFr,
                 '/api/automatic_speech_recognition/asr_crdnn_commonvoice_fr')
api.add_resource(ApiAsrCrdnnCommonvoiceDe,
                 '/api/automatic_speech_recognition/asr_crdnn_commonvoice_de')
api.add_resource(ApiAsrWav2vec2CommonvoiceIt,
                 '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_it')
api.add_resource(ApiAsrWav2vec2CommonvoiceFr,
                 '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_fr')
api.add_resource(ApiAsrWav2vec2CommonvoiceEn,
                 '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_en')
api.add_resource(ApiAsrWav2vec2CommonvoiceRw,
                 '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_rw')

api.add_resource(ApiAsrWav2vec2VoxpopuliDe,
                 '/api/automatic_speech_recognition/asr_wav2vec2_voxpopuli_de' )
api.add_resource(ApiAsrWav2vec2VoxpopuliEn,
                 '/api/automatic_speech_recognition/asr_wav2vec2_voxpopuli_en' )
api.add_resource(ApiAsrWav2vec2VoxpopuliEs,
                 '/api/automatic_speech_recognition/asr_wav2vec2_voxpopuli_es' )
api.add_resource(ApiAsrWav2vec2VoxpopuliFr,
                 '/api/automatic_speech_recognition/asr_wav2vec2_voxpopuli_fr' )
api.add_resource(ApiAsrWav2vec2VoxpopuliIt,
                 '/api/automatic_speech_recognition/asr_wav2vec2_voxpopuli_it' )

api.add_resource(ApiAsrWav2vec2TransformerAishellMandarinChinese,
                 '/api/automatic_speech_recognition/asr_wav2vec2_transformer_aishell_mandarin_chinese')
api.add_resource(ApiAsrConformerTransformerlmLibrispeechEn,
                 '/api/automatic_speech_recognition/asr_conformer_transformerlm_librispeech_en')
api.add_resource(ApiAsrCrdnntransformerlmLibrispeechEn,
                 '/api/automatic_speech_recognition/asr_crdnntransformerlm_librispeech_en')
api.add_resource(ApiAsrCrdnnrnnlmLibrispeechEn, '/api/automatic_speech_recognition/asr_crdnnrnnlm_librispeech_en' )
# ----------------------------------------------------------------------------------------------------------------------


# - DEEP LEARNING LANGUAGE IDENTIFICATION -------------------------------------------------------------------------
api.add_resource(ApiLangidEcapaCommonlanguage, '/api/language_id/langid_commonlanguage_ecapa')
api.add_resource(ApiLangidEcapaVoxLingua107, '/api/language_id/langid_voxlingua107_ecapa')
api.add_resource(ApiLangidToAsr, '/api/language_id/langid_asr')
# ----------------------------------------------------------------------------------------------------------------------

# - DEEP LEARNING VOICE ACTIVITY DETECTION -------------------------------------------------------------------------
api.add_resource(ApiVadCrdnnLibripartyCleaned,
                 '/api/voice_activity_detection/vad_crdnn_libriparty')

api.add_resource(ApiVadCrdnnLibripartyCleanedDownload,
                 '/api/voice_activity_detection/vad_crdnn_libriparty/<filename>')
# ----------------------------------------------------------------------------------------------------------------------

api.add_resource(ApiEmotionRecognitionWav2vec2IEMOCAP,
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
    app.run(debug=True, port=65000)