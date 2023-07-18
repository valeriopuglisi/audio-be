import os.path

from speechbrain.pretrained import VAD
import torchaudio
import torch
from IPython.display import Audio, display
import matplotlib.pyplot as plt
from pydub import AudioSegment


def print_stats(waveform, sample_rate=None, src=None):
  if src:
    print("-" * 10)
    print("Source:", src)
    print("-" * 10)
  if sample_rate:
    print("Sample Rate:", sample_rate)
  print("Shape:", tuple(waveform.shape))
  print("Dtype:", waveform.dtype)
  print(f" - Max:     {waveform.max().item():6.3f}")
  print(f" - Min:     {waveform.min().item():6.3f}")
  print(f" - Mean:    {waveform.mean().item():6.3f}")
  print(f" - Std Dev: {waveform.std().item():6.3f}")
  print()
  print(waveform)
  print()


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].plot(time_axis, waveform[c], linewidth=1)
    axes[c].grid(True)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
    if ylim:
      axes[c].set_ylim(ylim)
  figure.suptitle(title)
  plt.show(block=True)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
  waveform = waveform.numpy()

  num_channels, num_frames = waveform.shape
  time_axis = torch.arange(0, num_frames) / sample_rate

  figure, axes = plt.subplots(num_channels, 1)
  if num_channels == 1:
    axes = [axes]
  for c in range(num_channels):
    axes[c].specgram(waveform[c], Fs=sample_rate)
    if num_channels > 1:
      axes[c].set_ylabel(f'Channel {c+1}')
    if xlim:
      axes[c].set_xlim(xlim)
  figure.suptitle(title)
  plt.show(block=True)


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


audio_filename = 'valid.wav'
audio_name = audio_filename.split('.')[0]
results_path = 'pretrained_models/vad-crdnn-libriparty/'
audio_file = os.path.join(results_path, audio_filename)

VAD = VAD.from_hparams(source="speechbrain/vad-crdnn-libriparty", savedir="pretrained_models/vad-crdnn-libriparty")
# 1- Let's compute frame-level posteriors first
prob_chunks = VAD.get_speech_prob_file(audio_file)
# 2- Let's apply a threshold on top of the posteriors
prob_th = VAD.apply_threshold(prob_chunks).float()
# 3- Let's now derive the candidate speech segments
boundaries = VAD.get_boundaries(prob_th)
# 4- Apply energy VAD within each candidate speech segment (optional)
boundaries = VAD.energy_VAD(audio_file, boundaries)
# 5- Merge segments that are too close
boundaries = VAD.merge_close_segments(boundaries, close_th=0.250)
# 6- Remove segments that are too short
boundaries = VAD.remove_short_segments(boundaries, len_th=0.250)
# 7- Double-check speech segments (optional).
boundaries = VAD.double_check_speech_segments(boundaries, audio_file,  speech_th=0.5)
print("boundaries type:{}, boundaries.shape:{}, boundaries:{}".format(type(boundaries), boundaries.shape, boundaries))
waveform, sample_rate = torchaudio.load(audio_file)
print("waveform: {}".format(waveform))
final_waveform = torch.zeros(0)
for i, boundarie in enumerate(boundaries.numpy()):
    start_cut_time = round(boundarie[0] * sample_rate)
    end_cut_time = round(boundarie[1] * sample_rate)
    extracted_waveform = waveform[:, start_cut_time: end_cut_time]
    extracted_waveform = extracted_waveform.squeeze(0)
    final_waveform = torch.cat((final_waveform, extracted_waveform), 0)

output_filename = os.path.join(results_path, audio_name +'_result.wav')
final_waveform = final_waveform.unsqueeze(0)
torchaudio.save(output_filename, final_waveform, 16000)

