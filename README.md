# Negotiation Data Collector

This script collects conversational data for training a negotiation tactic detection model.

## Requirements

- Python 3.8+
- Hugging Face account with token for pyannote.audio

## Installation

1. Clone or download the script.
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variable: `export HF_TOKEN=your_huggingface_token`

## Usage

Run the script: `python negotiation_data_collector.py`

It will record 60 seconds of audio, transcribe it, perform diarization, and save the structured data to `session_[TIMESTAMP].json`.

## Output

The JSON file contains a list of turns, each with speaker_id, text, start_time, end_time, and an empty labels list for annotation.

## Error Handling

- If no speech is detected, the script will notify and exit.
- If HF_TOKEN is missing, diarization will fail with an error message.