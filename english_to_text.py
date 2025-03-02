import os
import whisper

# Load Whisper model for transcription
whisper_model = whisper.load_model("small")

# Transcribe audio from video
def transcribe_audio(audio_path, whisper_model):
    result = whisper_model.transcribe(audio_path)
    os.remove(audio_path)
    return result["text"]

if __name__ == "__main__":
    audio_path = "output_audio.wav"   
    whisper_model = whisper.load_model("small")
    original_transcription = transcribe_audio(audio_path, whisper_model)

    print(original_transcription)
