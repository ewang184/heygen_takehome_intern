from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
import numpy as np
import torch
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import resample_poly

sampling_rate, example_speech = wavfile.read("german_voice.wav")
if example_speech.dtype != np.float32:
    example_speech = example_speech.astype(np.float32) / np.iinfo(example_speech.dtype).max

if sampling_rate != 16000:
    example_speech = resample_poly(example_speech, 16000, sampling_rate)
    sampling_rate = 16000

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

speaker_embeddings = np.load("./spk_encode/output_audio.npy")
speaker_embeddings = torch.tensor(speaker_embeddings)
print(speaker_embeddings.shape)
#speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
#if speaker_embeddings.dim() == 3:  # Fix shape if it has an extra dimension
#    speaker_embeddings = speaker_embeddings.squeeze(1)

speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
