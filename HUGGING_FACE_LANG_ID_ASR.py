import matplotlib.pyplot as plt
from speechbrain.dataio.dataio import read_audio
from IPython.display import Audio
from speechbrain.pretrained import EncoderClassifier
import os
from deep_learning_dict_lang_id_to_asr import lang_to_asr

datset_path = 'media'
src_audio_name = 'common_voice_it_17415771.wav'
src = os.path.join(datset_path, src_audio_name)

# --------------------------- speechbrain/lang-id-voxlingua107-ecapa ---------------------------------------------

model_path = os.path.join('pretrained_models', 'lang-id-voxlingua107-ecapa')
language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=model_path)
signal = language_id.load_audio(src)
prediction = language_id.classify_batch(signal)
label = prediction[3]
lang = label[0].split(":")[0]

# -----------------------------------------------------------------------------------------------------------------
print("Language identified: {}".format(lang))
transcribed = lang_to_asr[lang](src)
print(transcribed)

