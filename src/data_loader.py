import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

# Mapping from CaSino annotations to target strategies
tactic_mapping = {
    'small-talk': 'Small Talk',
    'empathy': 'Empathy',
    'no-need': 'No-need',
    'vouching': 'Vouching',
    'non-hostile need': 'Non-hostile need',
    'firmness': 'Firmness',
    'elicit-pref': 'Elicit-Pref',
    'walk away': 'Walk Away'
}

# Target strategy list
strategy_cols = [
    'Small Talk',
    'Empathy',
    'No-need',
    'Vouching',
    'Non-hostile need',
    'Firmness',
    'Elicit-Pref',
    'Walk Away'
]
num_tactics = len(strategy_cols)
tactic_to_id = {tactic: i for i, tactic in enumerate(strategy_cols)}

class SlidingWindowProcessor:
    def __init__(self, window_size=5):
        self.window_size = window_size

    def process_dialogue(self, dialogue):
        """
        Takes a full dialogue (list of turns) and returns a list of windows.
        Each turn is a dict with 'speaker_id', 'text_content', 'annotations' (list of strings)
        Each window contains exactly window_size turns.
        """
        windows = []
        for i in range(len(dialogue) - self.window_size + 1):
            window = dialogue[i:i + self.window_size]
            windows.append(window)
        return windows

class TacticDataset(Dataset):
    def __init__(self, dataset_split='train', tokenizer_name='xlm-roberta-base', max_length=64):
        full_dataset = load_dataset('kchawla123/casino', split='train')
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        self.dataset = split_dataset['train'] if dataset_split == 'train' else split_dataset['test']
        print(f"Loaded '{dataset_split}' split with {len(self.dataset)} samples")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.sliding_window = SlidingWindowProcessor(window_size=5)
        self.strategy_cols = strategy_cols
        self.tactic_to_id = tactic_to_id
        print(f"Initialized TacticDataset for {dataset_split} with {len(self.strategy_cols)} labels.")
        self.all_windows = self._prepare_windows()

    def _prepare_windows(self):
        all_windows = []
        for sample in self.dataset:
            dialogue = self._extract_dialogue(sample)
            windows = self.sliding_window.process_dialogue(dialogue)
            all_windows.extend(windows)
        return all_windows

    def _extract_dialogue(self, sample):
        chat_logs = sample['chat_logs']
        annotations = sample['annotations']
        # Assume annotations align with chat_logs, pad if necessary
        dialogue = []
        for i, turn in enumerate(chat_logs):
            speaker_id = turn['id']
            text_content = turn['text']
            # Map speaker_id to int
            speaker_num = 0 if 'agent_1' in speaker_id else 1
            # Get annotations for this turn
            turn_annotations = annotations[i] if i < len(annotations) else []
            # Create multi-hot label_vector
            label_vector = [0] * len(self.strategy_cols)
            for ann in turn_annotations:
                if ann in tactic_mapping:
                    tactic = tactic_mapping[ann]
                elif ann in self.strategy_cols:
                    tactic = ann
                else:
                    tactic = None
                if tactic is not None and tactic in self.tactic_to_id:
                    label_vector[self.tactic_to_id[tactic]] = 1
            dialogue.append({
                'speaker_id': speaker_num,
                'text_content': text_content,
                'label_vector': label_vector
            })
        return dialogue

    def __len__(self):
        return len(self.all_windows)

    def __getitem__(self, idx):
        window = self.all_windows[idx]
        # Tokenize each turn
        input_ids = []
        attention_masks = []
        speaker_ids = []
        labels = []
        for turn in window:
            encoded = self.tokenizer(
                turn['text_content'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'].squeeze())
            attention_masks.append(encoded['attention_mask'].squeeze())
            speaker_ids.append(turn['speaker_id'])
            labels.append(turn['label_vector'])

        # Stack to tensors
        input_ids = torch.stack(input_ids)  # [window_size, seq_len]
        attention_mask = torch.stack(attention_masks)  # [window_size, seq_len]
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)  # [window_size]
        labels = torch.tensor(labels, dtype=torch.float)  # [window_size, num_tactics]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'speaker_ids': speaker_ids,
            'labels': labels
        }