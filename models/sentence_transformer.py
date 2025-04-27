import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from transformers import GPTNeoModel, AutoTokenizer

class SentenceTransformer_Pretrained_Backbone(nn.Module):

    def __init__(self,
        backbone_type: str = "EleutherAI/gpt-neo-125M",
        pooling: str = "mean"
    ):
        super().__init__()
        
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_type)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone  = GPTNeoModel.from_pretrained(backbone_type)
        self.pooling   = pooling

    def forward(self, sentences: list[str]) -> torch.Tensor:
        
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        # Move inputs to same device as model
        for k, v in encoded.items():
            encoded[k] = v.to(self.backbone.device)

        # Forward pass through GPT-Neo
        outputs = self.backbone(**encoded)
        last_hidden = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

        if self.pooling == "mean":
            # Mask out padding tokens for mean pooling
            mask   = encoded.attention_mask.unsqueeze(-1)       # (batch_size, seq_len, 1)
            summed = (last_hidden * mask).sum(dim=1)             # (batch_size, hidden_dim)
            counts = mask.sum(dim=1).clamp(min=1)                # avoid div-by-zero
            embeddings = summed / counts                         # mean over non-pad
        else:
            # Last-token pooling (since GPT-Neo has no [CLS] token)
            embeddings = last_hidden[:, -1, :]                   # (batch_size, hidden_dim)

        return embeddings
