import os
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import MarianMTModel, MarianTokenizer
import whisper
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer

# Load Whisper model for transcription
whisper_model = whisper.load_model("small")

# Transcribe audio from video
def transcribe_audio(video_path, whisper_model):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    result = whisper_model.transcribe(audio_path)
    os.remove(audio_path)
    return result["text"]

if __name__ == "__main__":
    video_path = "Tanzania-2.mp4"
    whisper_model = whisper.load_model("small")
    original_transcription = transcribe_audio(video_path, whisper_model)

    print(original_transcription)
