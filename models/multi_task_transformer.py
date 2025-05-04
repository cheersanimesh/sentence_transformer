import torch
import torch.nn as nn
from models.sentence_transformer import SentenceTransformer

class MultiTaskSentenceTransformer(nn.Module):
    def __init__(self,
                 encoder,
                 hidden_size: int = 768,
                 num_classes_sentence_classification: int = 4,
                 num_ner_labels: int = 9,
                 pooling: str = "mean"):
        super().__init__()

        # Shared backbone + pooling ( Shared Feature Extractor)
        self.encoder = encoder

        # Task-specific heads
        self.classifier_A = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_classes_sentence_classification)
        )

        #Token level NER Head
        self.ner_head = nn.Linear(hidden_size, num_ner_labels)

    def forward(self, sentences: list[str]):

        # Encode
        embeddings, last_hidden = self.encoder(sentences)  # (batch_size, hidden_size)

        # Get logits for Task A (Sentence Classification)
        logits_sentence_classification = self.classifier_A(embeddings)

        # Get logits for Task B ( NER)
        logits_ner = self.ner_head(last_hidden)

        return logits_sentence_classification, logits_ner