import torch
import torch.nn as nn
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel

class TacticSenseModel(nn.Module):
    def __init__(self, num_labels=8):
        super(TacticSenseModel, self).__init__()
        # Load the base transformer
        self.transformer = XLMRobertaModel.from_pretrained("xlm-roberta-base")
        
        # Speaker context: 0 for Buyer, 1 for Seller
        self.speaker_embedding = nn.Embedding(2, 768) 
        
        # Final classification layer
        self.classifier = nn.Linear(768, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, speaker_ids):
        # input_ids: [batch, 5, 64]
        batch_size, num_turns, seq_len = input_ids.shape
        
        # Flatten turns to process through transformer: [batch * 5, 64]
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        
        outputs = self.transformer(input_ids=flat_input_ids, attention_mask=flat_attention_mask)
        
        # Extract CLS (index 0) and reshape back to: [batch, 5, 768]
        cls_tokens = outputs.last_hidden_state[:, 0, :]
        cls_tokens = cls_tokens.view(batch_size, num_turns, -1)
        
        # Add Speaker Context: [batch, 5, 768]
        speaker_emb = self.speaker_embedding(speaker_ids)
        
        # Dimensionality check: [batch, 5, 768] + [batch, 5, 768]
        combined_context = cls_tokens + speaker_emb
        
        logits = self.classifier(self.dropout(combined_context))
        return logits
