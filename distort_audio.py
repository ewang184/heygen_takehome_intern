
'''
we have the chunks that we need - we get our alignments and then just pick all of them into a data structure that doesnt allow dupes.
then we get what we want - timing alignments for our words.
that is, we have the time that certain words were said that we should try to match
we do this by coercing the german audio into the english audio timeframes.
that is, we specifically care about some keywords'''

import torch
import torchaudio
import librosa
import librosa.display
import soundfile as sf
def stretch_audio(input_file, stretch_factor):
    # Load the audio file
    y, sr = librosa.load(input_file, sr=None)

    # Apply time-stretching
    y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)

    # Save or plot the output if needed
    sf.write('stretched_audio.wav', y_stretched, sr)

    return y_stretched, sr


# Stretch factor > 1.0 will lengthen the audio, < 1.0 will shorten it
input_file = 'output_audio.wav'
stretch_factor = 0.8  # Stretching by 1.5x (lengthening)
stretched_audio, sample_rate = stretch_audio(input_file, stretch_factor)

