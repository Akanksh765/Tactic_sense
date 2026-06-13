import collections
import os
import re
import torch

import speech_recognition as sr
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

from src.model import TacticSenseModel
from src.predict import predict_tactic, get_final_verdict

MODEL_PATH = os.path.join("models", "tacticsense_v1.pt")
WINDOW_SIZE = 5
SPEAKER_LABELS = {0: "BUYER", 1: "SELLER"}


def load_model(device="cpu"):
    model = TacticSenseModel()
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model weights from {MODEL_PATH}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Using randomly initialized model.")
    model.eval()
    return model


def load_tokenizer():
    print("Loading tokenizer: xlm-roberta-base")
    return XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")


def pad_window(turns, speaker_ids, window_size=WINDOW_SIZE):
    padded_turns = ["" for _ in range(window_size - len(turns))] + list(turns)
    padded_speaker_ids = [0 for _ in range(window_size - len(speaker_ids))] + list(speaker_ids)
    return padded_turns, padded_speaker_ids


def parse_speaker_prefix(text):
    cleaned = (text or "").strip()
    match = re.match(r"^(buyer|seller)\s*[:\-]\s*(.+)$", cleaned, flags=re.I)
    if not match:
        return None, cleaned
    role, rest = match.groups()
    return (0 if role.lower() == "buyer" else 1), rest.strip()


def choose_speaker(default_speaker):
    prompt = f"Speaker [b=buyer/s=seller, Enter={SPEAKER_LABELS[default_speaker]}]: "
    choice = input(prompt).strip().lower()
    if choice.startswith("s"):
        return 1
    if choice.startswith("b"):
        return 0
    return default_speaker


def main():
    model = load_model()
    tokenizer = load_tokenizer()

    recognizer = sr.Recognizer()
    turns = collections.deque(maxlen=WINDOW_SIZE)
    speaker_ids = collections.deque(maxlen=WINDOW_SIZE)
    current_speaker = 0

    print("TacticSense audio listener started.")
    print("Press Ctrl+C to stop.")
    print("Say 'buyer:' or 'seller:' before a sentence, or choose the speaker after transcription.")

    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)

            while True:
                print("\nListening for the next turn...")
                audio = recognizer.listen(source)

                try:
                    transcription = recognizer.recognize_google(audio)
                    print(f"Transcription: {transcription}")
                except sr.UnknownValueError:
                    print("Could not understand audio. Try again.")
                    continue
                except sr.RequestError as e:
                    print(f"Google Speech Recognition request failed: {e}")
                    continue

                prefixed_speaker, transcription = parse_speaker_prefix(transcription)
                speaker_id = prefixed_speaker if prefixed_speaker is not None else choose_speaker(current_speaker)
                current_speaker = speaker_id

                turns.append(transcription)
                speaker_ids.append(speaker_id)

                padded_turns, padded_speaker_ids = pad_window(turns, speaker_ids)
                labels, scores = predict_tactic(padded_turns, padded_speaker_ids, model, tokenizer)
                verdict = get_final_verdict(labels, scores)

                print("\n--- TacticSense Prediction ---")
                print(f"Speaker: {SPEAKER_LABELS[speaker_id]} ({speaker_id})")
                print(f"Current window ({len(turns)}/{WINDOW_SIZE}):")
                for idx, (turn, sid) in enumerate(zip(padded_turns, padded_speaker_ids), start=1):
                    print(f"  {idx}. [speaker={sid}] {turn}")
                print(verdict)
                print("--- End prediction ---")
    except KeyboardInterrupt:
        print("\nStopping TacticSense audio listener.")


if __name__ == "__main__":
    main()
