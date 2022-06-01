from deep_learning_audio_features import *

AudioAnalysisAPI = {
    "/api/voice_activity_detection/vad_crdnn_libriparty": {
        "task": "Voice Activity Detection",
        "dataset": "LibryParty",
        "system": "CRDNN",
        "performance": "F-score=0.9477 (test)",
        "api": "/api/voice_activity_detection/vad_crdnn_libriparty",
        "function": vad_crdnn_libriparty_cleaned,
    },
    # "": {
    #         "task": "Automatic Speech Recognition",
    #         "dataset": "LibriSpeech (English)",
    #         "system": "wav2vec2",
    #         "performance": "WER=1.90% (test-clean)",
    #         "api": ""
    #
    #     },
    "/api/automatic_speech_recognition/asr_crdnntransformerlm_librispeech_en":  {
        "task": "Automatic Speech Recognition",
        "dataset": "LibriSpeech (English)",
        "system": "CRDNN + Transformer LM",
        "performance": "WER=8.51% (test-clean)",
        "api": "/api/automatic_speech_recognition/asr_crdnntransformerlm_librispeech_en",
        "function": asr__crdnn_transformerlm__librispeech_en
        },

    '/api/automatic_speech_recognition/asr_crdnnrnnlm_librispeech_en': {
        "task": "Automatic Speech Recognition",
        "dataset": "LibriSpeech (English)",
        "system": "CRDNN + RNN +LM",
        "performance": "WER=3.09% (test-clean)",
        "api": '/api/automatic_speech_recognition/asr_crdnnrnnlm_librispeech_en',
        "function": asr__crdnn_rnn_lm__librispeech_en,
        },
    '/api/automatic_speech_recognition/asr_conformer_transformerlm_librispeech_en': {
        "task": "Automatic Speech Recognition",
        "dataset": "LibriSpeech (English)",
        "system": "Conformer + Transformer LM",
        "performance": "WER=3.09% (test-clean)",
        "api": '/api/automatic_speech_recognition/asr_conformer_transformerlm_librispeech_en',
        "function": asr__conformer_transformer_lm__librispeech_en,
        },

    # "": {
    #     "task": "Automatic Speech Recognition	",
    #     "dataset": "LibriSpeech (English)",
    #     "system": "CNN + Transformer",
    #     "performance": "WER=2.46% (test-clean)",
    #     "api": ""
    #     },
    # "":  {
    #     "task": "Automatic Speech Recognition	",
    #     "dataset": "TIMIT",
    #     "system": "CRDNN + distillation",
    #     "performance": "PER=13.1% (test)",
    #     "api": ""
    #     },
    # "": {
    #     "task": "Automatic Speech Recognition	",
    #     "dataset": "TIMIT",
    #     "system": "wav2vec2 + CTC/Att.",
    #     "performance": "PER=8.04% (test)",
    #     "api": ""
    #     },
    '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_en': {
        "task": "Automatic Speech Recognition",
        "dataset": "CommonVoice (English)",
        "system": "wav2vec2 + CTC",
        "performance": "WER=15.69% (test)",
        "api": '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_en',
        "function": asr__wav2vec2__commonvoice_en
        },
    '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_fr':    {
        "task": "Automatic Speech Recognition",
        "dataset": "CommonVoice (French)",
        "system": "wav2vec2 + CTC",
        "performance": "WER=9.96% (test), Test CER=3.19",
        "api": '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_fr',
        "function": asr__wav2vec2__commonvoice_fr,
        },
    '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_it': {
        "task": "Automatic Speech Recognition",
        "dataset": "CommonVoice (Italian)",
        "system": "wav2vec 2.0 with CTC/Attention",
        "performance": "WER=9.86% (test)",
        "api": '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_it',
        "function": asr__wav2vec2__commonvoice_it
        },
    '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_rw': {
        "task": "Automatic Speech Recognition",
        "dataset": "CommonVoice (Kinyarwanda)",
        "system": "wav2vec2 + seq2seq",
        "performance": "WER=18.91% (test)",
        "api": '/api/automatic_speech_recognition/asr_wav2vec2_commonvoice_rw',
        "function": asr__wav2vec2__commonvoice_rw
        },
    '/api/automatic_speech_recognition/asr_wav2vec2_transformer_aishell_mandarin_chinese': {
        "task": "Automatic Speech Recognition",
        "dataset": "AISHELL (Mandarin)",
        "system": "wav2vec2 + seq2seq",
        "performance": "CER=5.58% (test)",
        "api": '/api/automatic_speech_recognition/asr_wav2vec2_transformer_aishell_mandarin_chinese',
        "function": asr__wav2vec2_transformer__aishell_mandarin_chinese
        },
    # "":{
    #         "task": "Speech Translation",
    #         "dataset": "Fisher-callhome (spanish)",
    #         "system": "conformer (ST + ASR)",
    #         "performance": "BLEU=48.04 (test)",
    #         "api": ""
    #     },
    # "": {
    #         "task": "Speaker Verification	",
    #         "dataset": "VoxCeleb2",
    #         "system": "ECAPA-TDNN",
    #         "performance": "EER=0.69% (vox1-test)",
    #         "api": ""
    #     },
    # "": {
    #         "task": "Speaker Diarization	",
    #         "dataset": "AMI",
    #         "system": "ECAPA-TDNN",
    #         "performance": "DER=3.01% (eval)",
    #         "api": "",
    #     },
    '/api/speech_enhancement/enhancement_metricganplus_voicebank': {
            "task": "Speech Enhancement",
            "dataset": "VoiceBank",
            "system": "MetricGAN+",
            "performance": "PESQ=3.08 (test)",
            "api": '/api/speech_enhancement/enhancement_metricganplus_voicebank',
            "function": enhancement_metricganplus_voicebank
        },
    '/api/speech_enhancement/enhancement_sepformer_whamr': {
            "task": "Speech Enhancement",
            "dataset": "WHAMR!",
            "system": "SepFormer",
            "performance": "SI-SNR= 10.59, PESQ=2.84 (test)",
            "api": '/api/speech_enhancement/enhancement_sepformer_whamr',
            "function": enhancement_sepformer_whamr
    },
    '/api/speech_enhancement/enhancement_sepformer_wham': {
            "task": "Speech Enhancement",
            "dataset": "WHAM!: WSJ0-Mix dataset with environmental noise and reverberation in 8k",
            "system": "SepFormer",
            "performance": "Test-Set SI-SNR	Test-Set= 14.35; PESQ=3.07 (test)",
            "performance": "Test-Set SI-SNR	Test-Set= 14.35; PESQ=3.07 (test)",
            "api": '/api/speech_enhancement/enhancement_sepformer_wham',
            "function": enhancement_sepformer_wham
    },
    '/api/speech_enhancement/enhancement_sepformer_whamr_16k': {
            "task": "Speech Enhancement",
            "dataset": "WHAMR!: WSJ0-Mix dataset with environmental noise and reverberation in 16k",
            "system": "SepFormer",
            "description": "This repository provides all the necessary tools to perform audio source separation with a "
                           "SepFormer model, implemented with SpeechBrain, and pretrained on WHAMR! dataset with 16k "
                           "sampling frequency, which is basically a version of WSJ0-Mix dataset with environmental "
                           "noise and reverberation in 16k. For a better experience we encourage you to learn more "
                           "about SpeechBrain. The given model performance is 13.5 dB SI-SNRi on the test set of WHAMR!"
                           " dataset.",
            "performance": "Test-Set SI-SNRi 13.5 dB, Test-Set SDRi= 13.0 dB",
            "api": '/api/speech_enhancement/enhancement_sepformer_whamr_16k',
            "function": enhancement_sepformer_whamr_16k
    },

    '/api/audioseparation/speech_separation_sepformer_wsj02mix': {
            "task": "Speech Separation",
            "dataset": "WSJ2MIX",
            "system": "SepFormer",
            "performance": "SDRi=22.6 dB (test)",
            "api": '/api/audioseparation/speech_separation_sepformer_wsj02mix',
            "function": speechseparation_sepformer_wsj02mix
        },
    '/api/audioseparation/speech_separation_sepformer_wsj03mix': {
            "task": "Speech Separation",
            "dataset": "WSJ3MIX",
            "system": "SepFormer",
            "performance": "SDRi=20.0 dB (test)",
            "api": '/api/audioseparation/speech_separation_sepformer_wsj03mix',
            "function": speechseparation_sepformer_wsj03mix

        },
    '/api/audioseparation/speech_separation_sepformer_wham': {
            "task": "Speech Separation",
            "dataset": "WHAM!",
            "system": "SepFormer",
            "performance": "SDRi= 16.4 dB (test)",
            "api": '/api/audioseparation/speech_separation_sepformer_wham',
            "function": speechseparation_sepformer_wham
        },
    '/api/audioseparation/speech_separation_sepformer_whamr': {
            "task": "Speech Separation",
            "dataset": "WHAMR!",
            "system": "SepFormer",
            "performance": "SDRi= 14.0 dB (test)",
            "api": '/api/audioseparation/speech_separation_sepformer_whamr',
            "function": speechseparation_sepformer_whamr
        },
    # "":    {
    #         "task": "Speech Separation",
    #         "dataset": "Libri2Mix",
    #         "system": "SepFormer",
    #         "performance": "SDRi= 20.6 dB (test-clean)",
    #         "api": ""
    #
    #     },
    # "":  {
    #         "task": "Speech Separation",
    #         "dataset": "Libri3Mix",
    #         "system": "SepFormer",
    #         "performance": "SDRi= 18.7 dB (test-clean)",
    #         "api": ""
    #     },
    '/api/emotion_recognition/wav2vec2_IEMOCAP': {
            "task": "Emotion Recognition",
            "dataset": "IEMOCAP",
            "system": "wav2vec",
            "performance": "Accuracy=79.8% (test)",
            "api": '/api/emotion_recognition/wav2vec2_IEMOCAP',
            "function": emotion_recognition__wav2vec2__iemocap
    },
    '/api/language_id/langid_commonlanguage_ecapa': {
        "task": "Language Identification",
        "dataset": "CommonLanguage",
        "system": "ECAPA-TDNN",
        "performance": "Accuracy=84.9% (test)",
        "api": '/api/language_id/langid_commonlanguage_ecapa',
        "function": language_identification__ecapa__commonlanguage
    },
    '/api/language_id/langid_voxlingua107_ecapa': {
        "task": "Language Identification",
        "dataset": "VoxLingua 107",
        "system": "ECAPA-TDNN Sentence",
        "performance": "Accuracy=93.3% (test)",
        "api": '/api/language_id/langid_voxlingua107_ecapa',
        "function": language_identification__ecapa__vox_lingua107
    },

    '/api/language_id/langid_asr': {
            "task": "Language Identification + Automatic Speech Recognition",
            "dataset": "VoxLingua 107 for lang id and ",
            "system": "ECAPA-TDNN Sentence + wav2vec 2.0 with CTC/Attention",
            "performance": "Accuracy=93.3% (test) for land id and Test WER= 9.86 for ASR",
            "api": '/api/language_id/langid_asr',
            "function": lang_id__to__asr
    },

    # "Spoken, Language Understanding": [
    #     {
    #         task: "Spoken, Language Understanding",
    #         dataset: "Timers and Such",
    #         system: "CRDNN Intent",
    #         performance: "Accuracy=89.2% (test)",
    #         api: ""
    #     },
    #     {
    #         task: "Spoken, Language Understanding",
    #         dataset: "SLURP",
    #         system: "CRDNN	Intent",
    #         performance: "Accuracy=87.54% (test)",
    #         api: ""
    #     },
    # ]
}
