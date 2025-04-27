import torch
import torch.nn as nn
from sentence_transformer import SentenceTransformer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self,
                 pretrained_model_name: str = "EleutherAI/gpt-neo-125M",
                 hidden_size: int = 768,
                 num_classes_A: int = 4,
                 num_classes_B: int = 2,
                 pooling: str = "mean"):
        super().__init__()
        # Shared backbone + pooling
        self.encoder = SentenceTransformer(
            pretrained_model_name=pretrained_model_name,
            pooling=pooling
        )
        # Task-specific heads
        self.classifier_A = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_classes_A)
        )
        self.classifier_B = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_classes_B)
        )

    def forward(self, sentences: list[str]):
        emb = self.encoder(sentences)  # (batch_size, hidden_size)
        logits_A = self.classifier_A(emb)
        logits_B = self.classifier_B(emb)
        return logits_A, logits_B