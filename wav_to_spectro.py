import numpy as np
import resampy
from scipy.io import wavfile
import pdb 
import mel_features
import params

def shorter_waveform_to_examples(data):
  """
  Compute the spectrogram for each short audios
  Input: short audio data
  Output: list of spectrograms in this short audio, eahch with params.EXAMPLE_WINDOW_SECONDS, hopped by params.EXAMPLE_HOP_SECONDS
  """

  # Compute log mel spectrogram features for each short audios 
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=params.SAMPLE_RATE,
      log_offset=params.LOG_OFFSET,
      window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=params.NUM_MEL_BINS, # here forced the num_mel_bins 
      lower_edge_hertz=params.MEL_MIN_HZ,
      upper_edge_hertz=params.MEL_MAX_HZ)

  #(data.shape[0]/params.SAMPLE_RATE*1000-25)/10+1 FRAMES x num_mel_bins

  # Frame features into examples
  # Each example is [100x513]->[100x64bins] (non-overlapping)
  features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS #frames every second 
  example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(params.EXAMPLE_HOP_SECONDS * features_sample_rate)) 
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length) 
  return log_mel_examples

def segment_long_audio(wav_file):
  """ segment the long audio into short audios, with duration of params.SHORT_AUDIO_WINDOW_LENGTH_MIN, 
  overlapped by params.SHORT_AUDIO_HOP_LENGTH_MIN 
  Input: original long audio wav file
  Output: list of its short audios 
  """
  sample_rate, wav_data = wavfile.read(wav_file) # single audio file 
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  data = wav_data / 32768.0  # Convert to [-1.0, +1.0] 

  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)

  if len(data)==0:
    return 0 

  # Resample to the 16000
  if sample_rate != params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

  # frame the long audio into shorter ones 
  data_example_window_length = params.SHORT_AUDIO_WINDOW_LENGTH_MIN * 60 * params.SAMPLE_RATE 
  data_example_hop_length = params.SHORT_AUDIO_HOP_LENGTH_MIN * 60 * params.SAMPLE_RATE 
  data_examples = mel_features.frame(
      data,
      window_length=data_example_window_length,
      hop_length=data_example_hop_length) 
  return data_examples 


def wavfile_to_examples(wav_file):
  sample_rate, wav_data = wavfile.read(wav_file)  
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  data = wav_data / 32768.0  # Convert to [-1.0, +1.0]

  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  if len(data)==0:
    return 0 

  if sample_rate != params.SAMPLE_RATE:
    data = resampy.resample(data, sample_rate, params.SAMPLE_RATE)

  # Compute log mel spectrogram features for each short audios (log FBANK)
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=params.SAMPLE_RATE,
      log_offset=params.LOG_OFFSET,
      window_length_secs=params.STFT_WINDOW_LENGTH_SECONDS,
      hop_length_secs=params.STFT_HOP_LENGTH_SECONDS,
      num_mel_bins=params.NUM_MEL_BINS, # here forced the num_mel_bins 
      lower_edge_hertz=params.MEL_MIN_HZ,
      upper_edge_hertz=params.MEL_MAX_HZ)


  features_sample_rate = 1.0 / params.STFT_HOP_LENGTH_SECONDS 
  example_window_length = int(round(params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
  example_hop_length = int(round(params.EXAMPLE_HOP_SECONDS * features_sample_rate)) 

  # added: zero pad the frame to expected frame number for each example log-mel FBANK
  if log_mel.shape[0]%params.NUM_FRAMES:
    pad_data = np.zeros((int(np.ceil(1.0*log_mel.shape[0]/params.NUM_FRAMES)*params.NUM_FRAMES),log_mel.shape[1]))
    pad_data[:log_mel.shape[0],:log_mel.shape[1]] = log_mel
    log_mel = pad_data
  ##

  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length) 
  return log_mel_examples
