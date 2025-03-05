import os
from pydub import AudioSegment
from moviepy.editor import VideoFileClip

def split_files(wav_file, mp4_file, times, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the audio (WAV) file
    audio = AudioSegment.from_wav(wav_file)
    
    # Load the video (MP4) file
    video = VideoFileClip(mp4_file)
    
    last_time = 0
    audio_chunks = []
    video_chunks = []

    for time in times:
        # Split the audio
        audio_chunk = audio[last_time * 1000: time * 1000]  # Convert time to ms
        audio_chunks.append(audio_chunk)

        # Split the video
        video_chunk = video.subclip(last_time, time)
        video_chunks.append(video_chunk)

        last_time = time

    # Save the split audio files
    for i, chunk in enumerate(audio_chunks):
        audio_path = os.path.join(output_dir, f"audio_chunk_{i + 1}.wav")
        chunk.export(audio_path, format="wav")

    # Save the split video files
    for i, chunk in enumerate(video_chunks):
        video_path = os.path.join(output_dir, f"video_chunk_{i + 1}.mp4")
        chunk.write_videofile(video_path, codec="libx264")

    print(f"Splitting complete! Files saved to {output_dir}")

wav_file = "output_resynthesized.wav"
mp4_file = "Tanzania-2.mp4"
times = [4.0411875, 12.3438125, 21.8066875, 24.6275625, 29.689125, 35.370875, 39.9323125, 44.7538125, 51.7159375, 59.8184375]
output_dir = "split_files"

split_files(wav_file, mp4_file, times, output_dir)
