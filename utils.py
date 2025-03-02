import os
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import MarianMTModel, MarianTokenizer
import whisper
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer

# Load Whisper model for transcription
whisper_model = whisper.load_model("small")

# Load translation model
src_lang = "en"
tgt_lang = "de"
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Load German Coqui TTS model for proper pronunciation
german_tts_model = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", gpu=False)

# Load YourTTS voice conversion model
voice_conversion_model = Synthesizer(tts_checkpoint="tts_models/multilingual/vits--yourtts", gpu=False)

# Transcribe audio from video
def transcribe_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path)
    result = whisper_model.transcribe(audio_path)
    os.remove(audio_path)
    return result["text"]

# Translate text from English to German
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# Generate German audio from translated text using German Coqui TTS
def text_to_speech(text, output_audio):
    german_tts_model.tts_to_file(text=text, file_path=output_audio)

# Apply voice conversion using YourTTS to match original speaker's voice
def voice_conversion(input_audio, output_audio, reference_audio):
    # YourTTS voice conversion with speaker embedding from reference audio
    converted_wav = voice_conversion_model.tts_to_file(
        text=open(input_audio).read(), 
        file_path=output_audio, 
        speaker_wav=reference_audio
    )

# Replace audio in video
def replace_audio_in_video(video_path, audio_path, output_path):
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)
    video.write_videofile(output_path, codec='libx264', audio_codec='aac')

if __name__ == "__main__":
    video_path = "input_video.mp4"
    original_transcription = transcribe_audio(video_path)
    translated_text = translate_text(original_transcription)

    german_audio_path = "german_audio.wav"
    converted_audio_path = "converted_audio.wav"
    output_video_path = "output_video.mp4"

    text_to_speech(translated_text, german_audio_path)

    # Use original video audio as reference for voice conversion
    reference_audio_path = "temp_audio.wav"
    voice_conversion(german_audio_path, converted_audio_path, reference_audio_path)

    replace_audio_in_video(video_path, converted_audio_path, output_video_path)

    print(f"Translation complete! Output video saved to {output_video_path}")

