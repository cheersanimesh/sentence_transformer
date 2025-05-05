import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from transformers import GPTNeoModel, AutoTokenizer, RobertaModel

class SentenceTransformer(nn.Module):

    def __init__(self,
        backbone_type = "FacebookAI/roberta-base",
        pooling = "mean"
    ):
        super().__init__()
        
        
        backbone_decoder_only = False  # variable to determine backbone is decoder only

        # Load the pre-trained backbone transformer models
        if backbone_type == "EleutherAI/gpt-neo-125M":
            self.backbone  = GPTNeoModel.from_pretrained(backbone_type)
            self.max_token_length= 1024
            backbone_decoder_only = True
        elif backbone_type == 'FacebookAI/roberta-base':
            self.backbone = RobertaModel.from_pretrained(backbone_type)
            self.max_token_length = 512
        else:
            raise NotImplementedError

        #initalize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_type)
        # Use the EOS token as the padding token
        
        ## Apply for only decoder only model
        if backbone_decoder_only:
            self.tokenizer.pad_token = self.tokenizer.eos_token # Use the end-of-sequence token as the padding token

        # Stride size for overlapping long sequences 
        self.stride = int(0.2 * self.max_token_length)
        # Store pooling type ('mean' or 'last' or 'cls')
        self.pooling   = pooling

    def forward(self, sentences: list[str]) -> torch.Tensor:
        
        #tokenize sentences
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_token_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_attention_mask=True,
        )

        #import ipdb ; ipdb.set_trace()
        # Move inputs to same device as model
        for k, v in encoded.items():
            encoded[k] = v.to(self.backbone.device)
        
        # Forward pass through the backbone
        outputs = self.backbone(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )
        #import ipdb ; ipdb.set_trace()
        
        # last_hidden_state: embeddings for each token in each sentence
        last_hidden = outputs.last_hidden_state  

        # Mapping from each chunk back to its original sentence index
        overflow_to_sample = encoded["overflow_to_sample_mapping"]
        batch_size = len(sentences)
        hidden_size = last_hidden.size(-1)
        device = last_hidden.device

        
        # Prepare accumulators for sum pooling
        sum_hidden = torch.zeros(batch_size, hidden_size, device=device)
        token_counts = torch.zeros(batch_size, device=device)
        
        # Iterate over each chunk and aggregate:
        for chunk_idx, sample_idx in enumerate(overflow_to_sample):
            mask = encoded["attention_mask"][chunk_idx].unsqueeze(-1)   # (L,1)
            # sum over tokens
            sum_hidden[sample_idx] += (last_hidden[chunk_idx] * mask).sum(dim=0)
            token_counts[sample_idx] += mask.sum()

        #import ipdb; ipdb.set_trace()

        if self.pooling == "mean":

            embeddings = sum_hidden / token_counts.unsqueeze(-1)

        elif self.pooling == 'last_token':

            last_chunks = {}
            for idx, sample in enumerate(overflow_to_sample):
                last_chunks[sample.item()] = idx
            # gather last hidden state of last real token
            emb_list = []
            for sample, chunk_idx in sorted(last_chunks.items()):
                # find how many real tokens in that chunk
                real_lens = encoded["attention_mask"][chunk_idx].sum().item()
                emb_list.append(last_hidden[chunk_idx, real_lens-1])
            embeddings = torch.stack(emb_list, dim=0)      
         
        elif self.pooling == "cls":
            # take the first token's hidden state of the *first* chunk per sentence
            first_chunks = {}
            for chunk_idx, sample_idx in enumerate(overflow_to_sample):
                # record only the first chunk 
                if sample_idx.item() not in first_chunks:
                    first_chunks[sample_idx.item()] = chunk_idx
            # gather [CLS] (position 0) for each sample in order
            emb_list = [
                last_hidden[chunk_idx, 0]
                for sample_idx, chunk_idx in sorted(first_chunks.items())
            ]
            embeddings = torch.stack(emb_list, dim=0)         
        
        else:
            raise NotImplementedError
        
        return embeddings, last_hidden
