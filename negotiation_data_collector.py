import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from whisper import load_model
from pyannote.audio import Pipeline
from huggingface_hub import login
import json
from datetime import datetime

def record_audio(duration=60, fs=16000):
    print(f"Recording audio for {duration} seconds...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.float32)
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

def save_audio(audio, fs, filename):
    wav.write(filename, fs, audio)

def transcribe_audio(filename, model):
    print("Transcribing audio...")
    result = model.transcribe(filename, verbose=False)
    segments = result['segments']
    print("Transcription complete.")
    return segments

def diarize_audio(filename, hf_token):
    print("Performing speaker diarization...")
    if not hf_token:
        raise ValueError("Hugging Face token is required for diarization.")
    login(hf_token)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline(filename)
    print("Diarization complete.")
    return diarization

def get_speaker_for_segment(start, end, diarization):
    max_overlap = 0
    speaker = None
    for turn, _, label in diarization.itertracks(yield_label=True):
        overlap_start = max(start, turn.start)
        overlap_end = min(end, turn.end)
        overlap = max(0, overlap_end - overlap_start)
        if overlap > max_overlap:
            max_overlap = overlap
            speaker = label
    return speaker

def group_into_turns(segments, diarization):
    turns = []
    current_turn = None
    for seg in segments:
        speaker = get_speaker_for_segment(seg['start'], seg['end'], diarization)
        if speaker is None:
            continue  # skip if no speaker
        if current_turn is None or current_turn['speaker_id'] != speaker:
            if current_turn:
                turns.append(current_turn)
            current_turn = {
                'speaker_id': speaker,
                'text': seg['text'],
                'start_time': seg['start'],
                'end_time': seg['end'],
                'labels': []
            }
        else:
            current_turn['text'] += ' ' + seg['text']
            current_turn['end_time'] = seg['end']
    if current_turn:
        turns.append(current_turn)
    return turns

def main():
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not set. Diarization will fail.")
    try:
        # Record
        audio = record_audio()
        save_audio(audio, 16000, 'temp.wav')
        # Transcribe
        model = load_model("medium")
        segments = transcribe_audio('temp.wav', model)
        if not segments:
            print("No speech detected in the audio.")
            return
        # Diarize
        diarization = diarize_audio('temp.wav', hf_token)
        # Group
        turns = group_into_turns(segments, diarization)
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.json"
        data = {"turns": turns}
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Data saved to {filename}")
        # Clean up
        os.remove('temp.wav')
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()