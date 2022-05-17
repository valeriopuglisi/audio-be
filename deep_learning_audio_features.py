from speechbrain.pretrained import SepformerSeparation as separator
from speechbrain.pretrained import *
from cfg import *
import librosa
import speechbrain.pretrained
import soundfile as sf
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.dataio.preprocess import AudioNormalizer

# ---------------------------- AUDIO SEPARATION ---------------------------------


def audioseparation_sepformer_whamr(audiofile_path):
    model_path = os.path.join('pretrained_models', 'sepformer-whamr')
    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir=model_path)
    est_sources = model.separate_file(path=audiofile_path)
    filename = os.path.split(audiofile_path)[-1]
    output_filename_1 = "AudioseparationSepformerWham_source1_" + filename
    output_filename_2 = "AudioseparationSepformerWham_source2_" + filename
    output_filename_1_path = os.path.join(AUDIO_SEPARATION_SEPFORMER_WHAMR, output_filename_1)
    output_filename_2_path = os.path.join(AUDIO_SEPARATION_SEPFORMER_WHAMR, output_filename_2)
    torchaudio.save(output_filename_1_path, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(output_filename_2_path, est_sources[:, :, 1].detach().cpu(), 8000)
    return [output_filename_1_path, output_filename_2_path]


def audioseparation_sepformer_wham(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'sepformer-wham')
    model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir=model_path)
    est_sources = model.separate_file(path=audiofile_path)
    filename = os.path.split(audiofile_path)[-1]
    output_filename_1 = "AudioseparationSepformerWham_source1_" + filename
    output_filename_2 = "AudioseparationSepformerWham_source2_" + filename
    output_filename_1_path = os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, output_filename_1)
    output_filename_2_path = os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, output_filename_2)
    torchaudio.save(output_filename_1_path, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(output_filename_2_path, est_sources[:, :, 1].detach().cpu(), 8000)
    return [output_filename_1_path, output_filename_2_path]

# ------------------------------------------------------------------------
# ---------------------------- AUDIO ENHANCEMENT ---------------------------------


def enhancement_sepformer_wham(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'sepformer-wham-enhancement')
    model = separator.from_hparams(source="speechbrain/sepformer-wham-enhancement", savedir=model_path)
    est_sources = model.separate_file(path=audiofile_path)
    filename = os.path.split(audiofile_path)[-1]
    output_filename = "EnhancementSepformerWham_" + filename
    output_filename_path = os.path.join(ENHANCEMENT_SEPFORMER_WHAM_DIR, output_filename)
    torchaudio.save(output_filename_path, est_sources[:, :, 0].detach().cpu(), 8000)
    return [output_filename_path]


def enhancement_sepformer_whamr(audiofile_path):
    """
        ** SepFormer trained on WHAMR! for speech enhancement (8k sampling frequency)
        This repository provides all the necessary tools to perform speech enhancement (denoising + dereverberation)
         with a SepFormer model, implemented with SpeechBrain, and pretrained on WHAMR! dataset with 8k sampling frequency,
         which is basically a version of WSJ0-Mix dataset with environmental noise and reverberation in 8k.
        The given model performance is 10.59 dB SI-SNR on the test set of WHAMR! dataset.
        Release	    |Test-Set SI-SNR	| Test-Set PESQ
        01-12-21	    10.59	                2.84
        """
    model_path = os.path.join('pretrained_models', 'sepformer-whamr-enhancement')
    model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir=model_path)
    est_sources = model.separate_file(path=audiofile_path)
    filename = os.path.split(audiofile_path)[-1]
    output_filename = "EnhancementSepformerWhamr_" + filename
    output_filename_path = os.path.join(ENHANCEMENT_SEPFORMER_WHAMR_DIR, output_filename)
    torchaudio.save(output_filename_path, est_sources[:, :, 0].detach().cpu(), 8000)
    return [output_filename_path]


def enhancement_metricganplus_voicebank(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'metricgan-plus-voicebank')
    model = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank", savedir=model_path)
    # Load and add fake batch dimension
    noisy = model.load_audio(audiofile_path).unsqueeze(0)
    # Add relative length tensor
    enhanced = model.enhance_batch(noisy, lengths=torch.tensor([1.]))
    # Saving enhanced signal on disk
    filename = os.path.split(audiofile_path)[-1]
    output_filename = "EnhancementMetricganplusVoicebank_" + filename
    output_filename_path = os.path.join(ENHANCEMENT_METRICGANPLUS_VOICEBANK_DIR, output_filename)
    torchaudio.save(output_filename_path, enhanced.cpu(), 16000)
    return [output_filename_path]
# ------------------------------------------------------------------------

# ---------------------------- SPEECH SEPARATION ---------------------------------


def speechseparation_sepformer_wham(audiofile_path):
    model_path = os.path.join('pretrained_models', 'sepformer-wham')
    model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir=model_path)
    est_sources = model.separate_file(path=audiofile_path)
    filename = os.path.split(audiofile_path)[-1]
    output_filename_1 = "SpeechSeparationSepformerWham_source1_" + filename
    output_filename_2 = "SpeechSeparationSepformerWham_source2_" + filename
    output_filename_1_path = os.path.join(SEPARATION_WHAM_DIR, output_filename_1)
    output_filename_2_path = os.path.join(SEPARATION_WHAM_DIR, output_filename_2)
    torchaudio.save(output_filename_1_path, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(output_filename_2_path, est_sources[:, :, 1].detach().cpu(), 8000)
    separated_file_paths = [output_filename_1, output_filename_2]
    return separated_file_paths


def speechseparation_sepformer_whamr(audiofile_path):
    model_path = os.path.join('pretrained_models', 'sepformer-whamr')
    model = separator.from_hparams(source="speechbrain/sepformer-whamr", savedir=model_path)
    est_sources = model.separate_file(path=audiofile_path)
    filename = os.path.split(audiofile_path)[-1]
    output_filename_1 = "ApiSpeechSeparationSepformerWhamr" + "_source1_" + filename
    output_filename_2 = "ApiSpeechSeparationSepformerWhamr" + "_source2_" + filename
    output_filename_1_path = os.path.join(SEPARATION_WHAMR_DIR, output_filename_1)
    output_filename_2_path = os.path.join(SEPARATION_WHAMR_DIR, output_filename_2)
    torchaudio.save(output_filename_1_path, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(output_filename_2_path, est_sources[:, :, 1].detach().cpu(), 8000)
    separated_file_paths = [output_filename_1_path, output_filename_2_path]
    return separated_file_paths


def speechseparation_sepformer_wsj02mix(audiofile_path):
    model_path = os.path.join('pretrained_models', 'sepformer-wsj02mix')
    model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir=model_path)
    est_sources = model.separate_file(path=audiofile_path)
    filename = os.path.split(audiofile_path)[-1]
    output_filename_1 = "SpeechSeparationSepformerWsj02mix_source1_" + filename
    output_filename_2 = "SpeechSeparationSepformerWsj02mix_source2_" + filename
    output_filename_1_path = os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, output_filename_1)
    output_filename_2_path = os.path.join(SEPARATION_SEPFORMER_WSJ2_DIR, output_filename_2)
    torchaudio.save(output_filename_1_path, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(output_filename_2_path, est_sources[:, :, 1].detach().cpu(), 8000)
    separated_filenames = [output_filename_1_path, output_filename_2]
    return separated_filenames


def speechseparation_sepformer_wsj03mix(audiofile_path):
    model_path = os.path.join('pretrained_models', 'sepformer-wsj03mix')
    model = separator.from_hparams(source="speechbrain/sepformer-wsj03mix", savedir=model_path)
    est_sources = model.separate_file(path=audiofile_path)
    filename = os.path.split(audiofile_path)[-1]
    output_filename_1 = "SpeechSeparationSepformerWsj03mix_source1_" + filename
    output_filename_2 = "SpeechSeparationSepformerWsj03mix_source2_" + filename
    output_filename_3 = "SpeechSeparationSepformerWsj03mix_source3_" + filename
    output_filename_1_path = os.path.join(SEPARATION_SEPFORMER_WSJ3_DIR, output_filename_1)
    output_filename_2_path = os.path.join(SEPARATION_SEPFORMER_WSJ3_DIR, output_filename_2)
    output_filename_3_path = os.path.join(SEPARATION_SEPFORMER_WSJ3_DIR, output_filename_3)
    torchaudio.save(output_filename_1_path, est_sources[:, :, 0].detach().cpu(), 8000)
    torchaudio.save(output_filename_2_path, est_sources[:, :, 1].detach().cpu(), 8000)
    torchaudio.save(output_filename_3_path, est_sources[:, :, 2].detach().cpu(), 8000)
    separated_filenames = [output_filename_1_path, output_filename_2_path, output_filename_3_path]
    return separated_filenames
# ----------------------------------------------------------------------------------------------------

# ---------------------------- VOICE ACTIVITY DETECTION ----------------------------------------------


def vad_crdnn_libriparty(audiofile_path):
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
    return lines


def vad_crdnn_libriparty_cleaned(audiofile_path):
    y, sr = librosa.load(audiofile_path)
    sr1 = 16000
    y1 = librosa.resample(y, orig_sr=sr, target_sr=sr1)
    filename = "16khz_" + os.path.split(audiofile_path)[-1]
    audio_path_16khz = os.path.join(MEDIA_DIR, filename)
    output_filename = "VAD_" + filename
    output_filename_path = os.path.join(VAD_CRDNN, output_filename)
    sf.write(audio_path_16khz, y1, sr1)

    model_path = os.path.join('pretrained_models', 'vad-crdnn-libriparty')
    # boundaries = VAD.get_speech_segments(audio_path_16khz)
    VAD = speechbrain.pretrained.VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir=model_path)
    # 1- Let's compute frame-level posteriors first
    prob_chunks = VAD.get_speech_prob_file(audio_path_16khz)
    # 2- Let's apply a threshold on top of the posteriors
    prob_th = VAD.apply_threshold(prob_chunks).float()
    # 3- Let's now derive the candidate speech segments
    boundaries = VAD.get_boundaries(prob_th)
    # 4- Apply energy VAD within each candidate speech segment (optional)
    boundaries = VAD.energy_VAD(audio_path_16khz, boundaries)
    # 5- Merge segments that are too close
    boundaries = VAD.merge_close_segments(boundaries, close_th=0.250)
    # 6- Remove segments that are too short
    boundaries = VAD.remove_short_segments(boundaries, len_th=0.250)
    # 7- Double-check speech segments (optional).
    boundaries = VAD.double_check_speech_segments(boundaries, audio_path_16khz, speech_th=0.5)

    waveform, sample_rate = torchaudio.load(audio_path_16khz)
    final_waveform = torch.zeros(0)
    for i, boundarie in enumerate(boundaries.numpy()):
        start_cut_time = round(boundarie[0] * sample_rate)
        end_cut_time = round(boundarie[1] * sample_rate)
        extracted_waveform = waveform[:, start_cut_time: end_cut_time]
        extracted_waveform = extracted_waveform.squeeze(0)
        final_waveform = torch.cat((final_waveform, extracted_waveform), 0)
    final_waveform = final_waveform.unsqueeze(0)
    torchaudio.save(output_filename_path, final_waveform, sample_rate)
    return [output_filename_path]
# ----------------------------------------------------------------------------------------

# ---------------------------- EMOTION RECOGNITION ---------------------------------------


def emotion_recognition__wav2vec2__iemocap(audiofile_path):

    model_path = os.path.join('pretrained_models', 'emotion-recognition-wav2vec2-IEMOCAP')
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                               pymodule_file="custom_interface.py",
                               classname="CustomEncoderWav2vec2Classifier",
                               savedir=model_path)
    out_prob, score, index, text_lab = classifier.classify_file(audiofile_path)
    return text_lab
# ------------------------------------------------------------------------


# ---------------------------- ASR ---------------------------------------

def asr__wav2vec2__commonvoice_fr(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'asr-wav2vec2-commonvoice-fr')
    asr_model = speechbrain.pretrained.EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-fr", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__wav2vec2__commonvoice_it(audiofile_path):
    model_path = os.path.join('pretrained_models', 'asr-wav2vec2-commonvoice-it')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-it",
                                               savedir="pretrained_models/asr-wav2vec2-commonvoice-it")

    transcribed = asr_model.transcribe_file(audiofile_path)


    return transcribed


def asr__wav2vec2__commonvoice_en(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'asr-wav2vec2-commonvoice-en')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__wav2vec2__commonvoice_rw(audiofile_path):
    """
       This ASR system is composed of 2 different but linked blocks:
        -- Tokenizer (unigram) that transforms words into subword units and trained with the train transcriptions
            (train.tsv) of CommonVoice (RW).
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
    model_path = os.path.join('pretrained_models', 'asr-wav2vec2-commonvoice-rw')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-rw", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__wav2vec2_transformer__aishell_mandarin_chinese(audiofile_path):
    model_path = os.path.join('pretrained_models', 'asr-wav2vec2-transformer-aishell')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-transformer-aishell", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__crdnn_transformerlm__librispeech_en(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'asr-crdnn-transformerlm-librispeech')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-transformerlm-librispeech", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__crdnn_rnn_lm__librispeech_en(audiofile_path):
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

    model_path = os.path.join('pretrained_models', 'asr-crdnn-rnnlm-librispeech')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-rnnlm-librispeech", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__crdnn__commonvoice_fr(audiofile_path):
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

    model_path = os.path.join('pretrained_models', 'asr-crdnn-commonvoice-fr')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-fr", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__crdnn__commonvoice_it(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'asr-crdnn-commonvoice-it')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-it", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__crdnn__commonvoice_de(audiofile_path):
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

    model_path = os.path.join('pretrained_models', 'asr-crdnn-commonvoice-de')
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-crdnn-commonvoice-de", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__conformer_transformer_lm__ksponspeech_korean(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'asr-conformer-transformerlm-ksponspeech')
    asr_model = EncoderDecoderASR.from_hparams(source="ddwkim/asr-conformer-transformerlm-ksponspeech", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file


def asr__conformer_transformer_lm__librispeech_en(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'asr-conformer-transformerlm-ksponspeech')
    asr_model = EncoderDecoderASR.from_hparams(source="ddwkim/asr-conformer-transformerlm-ksponspeech", savedir=model_path)
    transcribed_file = asr_model.transcribe_file(audiofile_path)
    return transcribed_file
# ------------------------------------------------------------------------


# ---------------------------- LANGUAGE IDENTIFICATION ---------------------------------------

def language_identification__ecapa__commonlanguage(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'lang-id-commonlanguage_ecapa')
    classifier = EncoderClassifier.from_hparams(source="speechbrain/lang-id-commonlanguage_ecapa", savedir=model_path)
    out_prob, score, index, text_lab = classifier.classify_file(audiofile_path)
    print(text_lab)
    return text_lab


def language_identification__ecapa__vox_lingua107(audiofile_path):
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
    model_path = os.path.join('pretrained_models', 'lang-id-voxlingua107-ecapa')
    language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=model_path)
    # Download Thai language sample from Omniglot and cvert to suitable form
    signal = language_id.load_audio(audiofile_path)
    prediction = language_id.classify_batch(signal)
    lang = prediction[3]
    return lang
# ------------------------------------------------------------------------
