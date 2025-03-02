import ffmpeg

def extract_audio(video_path, audio_path):
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, format='wav')
        .run()
    )

# Example usage
video_path = "Tanzania-2.mp4"
audio_path = "output_audio.wav"
extract_audio(video_path, audio_path)
print(f"Audio extracted to {audio_path}")

