'''import ffmpeg

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
print(f"Audio extracted to {audio_path}")'''



import ffmpeg
import torchaudio

def extract_audio(video_path: str, audio_path: str) -> None:
    """
    Extracts the audio from a video file, resamples it to 16000 Hz, and saves it as a WAV file.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the extracted and resampled audio file.
    """
    temp_audio_path = "temp_output_audio.wav"

    (
        ffmpeg
        .input(video_path)
        .output(temp_audio_path, format='wav')
        .run()
    )

    # Resample audio to 16000 Hz
    signal, fs = torchaudio.load(temp_audio_path)
    target_fs = 16000
    if fs != target_fs:
        resampler = torchaudio.transforms.Resample(orig_freq=fs, new_freq=target_fs)
        signal = resampler(signal)

    if signal.size(0) == 2:  # If stereo, convert to mono
        signal = signal.mean(dim=0, keepdim=True)

    torchaudio.save(audio_path, signal, target_fs)

if __name__ == "__main__":
    VIDEO_PATH = "Tanzania-2.mp4"
    AUDIO_PATH = "output_audio.wav"

    extract_audio(VIDEO_PATH, AUDIO_PATH)
    print(f"Audio extracted and resampled to {AUDIO_PATH}")


