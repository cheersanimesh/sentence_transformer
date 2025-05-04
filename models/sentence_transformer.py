import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import GPTNeoModel, AutoTokenizer

class SentenceTransformer(nn.Module):

    def __init__(self,
        backbone_type = "EleutherAI/gpt-neo-125M",
        pooling = "mean"
    ):
        super().__init__()
        
        #initalize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_type)

        self.tokenizer.pad_token = self.tokenizer.eos_token # Use the end-of-sequence token as the padding token

        # Load the pre-trained GPT-Neo transformer model
        self.backbone  = GPTNeoModel.from_pretrained(backbone_type)

        # Store pooling type ('mean' or 'last')
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

        # last_hidden_state: embeddings for each token in each sentence
        last_hidden = outputs.last_hidden_state  

        if self.pooling == "mean":
            # Mask out padding tokens for mean pooling
            mask   = encoded.attention_mask.unsqueeze(-1)      
            # Sum token embeddings where mask == 1
            summed = (last_hidden * mask).sum(dim=1)             
            # Count non-padded tokens per sentence, avoid division by zero
            counts = mask.sum(dim=1).clamp(min=1)            
            ## mean over non-pad
            embeddings = summed / counts                         
        else:
            # Last-token pooling (since GPT-Neo has no [CLS] token)
            embeddings = last_hidden[:, -1, :]                   

        return embeddings, last_hidden
