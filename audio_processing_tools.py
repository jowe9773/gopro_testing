#audio_processing_tools.py
"""module with audio processing tools"""

import tempfile
import numpy as np
import moviepy.editor as mp
from scipy.signal import correlate
from scipy.io import wavfile
from file_managers import FileManagers
import multiprocessing

def extract_audio(video_path):
    print(video_path)
    clip = mp.VideoFileClip(video_path)
    audio = clip.audio
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
    rate, audio_data = wavfile.read(temp_audio_path)
    return rate, audio_data

def find_time_offset(rate1, audio1, rate2, audio2):
    # Ensure the sample rates are the same
    if rate1 != rate2:
        raise ValueError("Sample rates of the two audio tracks do not match")

    # Convert stereo to mono if necessary
    if len(audio1.shape) == 2:
        audio1 = audio1.mean(axis=1)
    if len(audio2.shape) == 2:
        audio2 = audio2.mean(axis=1)

    # Normalize audio data to avoid overflow
    audio1 = audio1 / np.max(np.abs(audio1))
    audio2 = audio2 / np.max(np.abs(audio2))

    # Compute cross-correlation
    correlation = correlate(audio1, audio2)
    lag = np.argmax(correlation) - len(audio2) + 1

    # Calculate the time offset in seconds
    time_offset = lag / rate1

    return time_offset

def find_time_offset_wrapper(params):
    return find_time_offset(*params)

if __name__ == '__main__':
    fm = FileManagers()

    video_files = [fm.load_fn("Select video1"), fm.load_fn("Select video2"), fm.load_fn("Select video3"), fm.load_fn("Select video4")]

    # Step 1: Extract audio from videos using multiprocessing
    with multiprocessing.Pool(processes=len(video_files)) as pool:
        audio_results_async = [pool.apply_async(extract_audio, (video,)) for video in video_files]
        audio_results = [res.get() for res in audio_results_async]

    rates, audios = zip(*audio_results)

    # Step 2: Prepare parameters for find_time_offset
    audio_pairs = [(rates[0], audios[0], rates[i], audios[i]) for i in range(1, len(rates))]

    print(audio_pairs)