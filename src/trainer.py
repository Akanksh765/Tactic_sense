import os
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from src.data_loader import TacticDataset
from src.model import TacticSenseModel

# Added num_epochs=4 as a default parameter here
def train(num_epochs=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_dataset = TacticDataset(dataset_split='train')
    test_dataset = TacticDataset(dataset_split='test')

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = TacticSenseModel().to(device)
    criterion = BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_val_loss = float('inf')
    # Removed the hardcoded num_epochs = 4 line from here
    save_path = os.path.join('models', 'tacticsense_v1.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            speaker_ids = batch['speaker_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, speaker_ids=speaker_ids)
            target = labels[:, -1, :]
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

        avg_train_loss = train_loss / max(train_steps, 1)

        model.eval()
        val_loss = 0.0
        val_steps = 0
        exact_matches = 0
        total_examples = 0

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids=input_ids, attention_mask=attention_mask, speaker_ids=speaker_ids)
                target = labels[:, -1, :]
                loss = criterion(logits, target)

                val_loss += loss.item()
                val_steps += 1

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                exact_matches += (preds == target.long()).all(dim=1).sum().item()
                total_examples += preds.size(0)

        avg_val_loss = val_loss / max(val_steps, 1)
        exact_match_ratio = exact_matches / max(total_examples, 1)

        print(
            f'Epoch {epoch}/{num_epochs} | '
            f'Train Loss: {avg_train_loss:.4f} | '
            f'Val Loss: {avg_val_loss:.4f} | '
            f'Exact Match Ratio: {exact_match_ratio:.4f}'
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f'Saved improved model to {save_path}')


if __name__ == '__main__':
    train()