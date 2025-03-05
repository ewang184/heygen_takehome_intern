import os
from pydub import AudioSegment
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from tqdm import tqdm
import librosa
import soundfile as sf

def warp_audio(input_path, output_path, target_duration):
    # Load audio
    y, sr = librosa.load(input_path, sr=None)
    y = y.astype('float32')
    # Get current duration
    current_duration = librosa.get_duration(y=y, sr=sr)
    
    # Calculate time stretch factor
    stretch_factor = current_duration / target_duration
    
    # Stretch the audio
    try:
        y_stretched = librosa.effects.time_stretch(y, rate=stretch_factor)
    except Exception as e:
        print(f"Error during time stretching: {e}")
        print(f"y dtype: {y.dtype}, y shape: {y.shape}, sr: {sr}, stretch factor: {stretch_factor}")
        raise
    # Save the warped audio
    sf.write(output_path, y_stretched, sr)

def process_text(text):
    # Remove all newlines and extra spaces
    cleaned_text = text.replace("\n", " ").replace("\r", " ").strip()

    # Split the text into sentences based on periods
    sentences = [sentence.strip() for sentence in cleaned_text.split(".") if sentence.strip()]

    return sentences

def combine_wavs_from_directory(directory, output_file):
    # Initialize an empty AudioSegment
    combined = AudioSegment.empty()

    # Get all .wav files in the directory
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    
    # Sort the files to ensure they are combined in the correct order
    wav_files.sort()

    # Loop through the list of wav files and append them
    for i,wav_file in enumerate(wav_files):
        wav_path = os.path.join(directory, wav_file)
        audio = AudioSegment.from_wav(wav_path)

        if i > 0:
            silence = AudioSegment.silent(duration=200)  # Create silence of the specified duration
            combined += silence

        combined += audio

    # Export the combined audio to a single file
    combined.export(output_file, format="wav")
    print(f"Combined audio saved to {output_file}")

def combine_wavs_from_directory_crossfade(directory, output_file, crossfade_duration=1000):
    # Initialize an empty AudioSegment
    combined = AudioSegment.empty()

    # Get all .wav files in the directory
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    # Sort the files to ensure they are combined in the correct order
    wav_files.sort()

    # Loop through the list of wav files and append them
    for i, wav_file in enumerate(wav_files):
        wav_path = os.path.join(directory, wav_file)
        audio = AudioSegment.from_wav(wav_path)
        
        # Apply crossfade if it's not the first file
        if i > 0:
            combined = combined.append(audio, crossfade=crossfade_duration)
        else:
            combined += audio
    
    # Export the combined audio to a single file
    combined.export(output_file, format="wav")
    print(f"Combined audio saved to {output_file}")

def get_difference(times):
    values = [0]+times
    differences = [values[i] - values[i - 1] for i in range(1, len(values))]
    return differences

def combine_wavs_given_time(directory, output_file, times):
    differences = get_difference(times)
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    wav_files.sort()
    for i,time in enumerate(differences):
        warp_audio(os.path.join("german_sentences", wav_files[i]), f"./warped_audio/{i}warped.wav", target_duration=time)        

    combine_wavs_from_directory("warped_audio", output_file)

text = """
    Tansania, die Heimat einiger der atemberaubendsten Tierwelten der Erde. 
    Hier im Herzen Ostafrikas, beherbergt der große Serengeti-Nationalpark eines der größten Spektakel der Natur, die Große Migration. Über eine Million Wildebeest, Zebras und Gazellen reisen weite Strecken auf der Suche nach frischem Gras, glühende Flüsse voller Krokodile. 
    Aber Raubtiere sind nie weit dahinter. Löwen, die Könige der Savanne, verfolgen ihre Beute mit Geduld und Präzision. 
    Cheetahs, die schnellsten Landtiere, jagen ihre Ziele in einer spannenden Schau der Geschwindigkeit. 
    Im üppigen Terengair-Nationalpark streifen riesige Elefanten frei umher. 
    Diese intelligenten Kreaturen bilden starke Familienbande, die ihre Jungen vor Bedrohungen schützen. 
    Und im alten Krater Ngorongoro findet der gefährdete schwarze Nashorn Zuflucht, ein seltener Anblick in der Wildnis. 
    Tansanias Wildtiere sind ein Schatz wie kein anderer, ein zartes Gleichgewicht der Natur, das uns an die Schönheit und Kraft der Wildnis erinnert.
    """

'''tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

sentences = process_text(text)

for i, sentence in tqdm(enumerate(sentences), total=len(sentences), desc="Generating Speech"):
    tts.tts_to_file(text=sentence,
                    file_path=f"german_sentences/{i}german.wav",
                    speaker_wav="output_audio.wav",
                    language="de")'''

sentence_times = [4.0411875, 12.3438125, 21.8066875, 24.6275625, 29.689125, 35.370875, 39.9323125, 44.7538125, 51.7159375, 59.8184375]

#combine_wavs_from_directory('german_sentences', 'german_voice.wav')

B
combine_wavs_given_time('german_sentences', 'warped_german_voice.wav', sentence_times)
