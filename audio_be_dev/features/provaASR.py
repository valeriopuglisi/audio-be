from speechbrain.pretrained import EncoderDecoderASR
import os

datset_path = 'media'
src_audio_name = 'common_voice_it_17415771.wav'
src = os.path.join(datset_path, src_audio_name)

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-it",
                                           savedir="pretrained_models/asr-wav2vec2-commonvoice-it")

transcribed = asr_model.transcribe_file(src)
print(transcribed)