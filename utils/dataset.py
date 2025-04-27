import json
import torch
from torch.utils.data import Dataset

import json
import torch
from torch.utils.data import Dataset

class DummyMTLDataset(Dataset):
    def __init__(self, json_path, sent2idx, ent2idx):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.sent2idx = sent2idx
        self.ent2idx = ent2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        sentence = entry["sentence"]
        cls_label = self.sent2idx[entry["task_A"]]

        # build a simple list of span‐dicts so we can align inside the model
        spans = entry["task_B"]

        return sentence, cls_label, spans


def collate_fn(batch):
    sentences, cls_labels, spans = zip(*batch)
    return {
        "sentences":     list(sentences),
        "cls_labels":    torch.tensor(cls_labels, dtype=torch.long),
        "spans":         list(spans),
    }


# class DummyMTLDataset(Dataset):
#     def __init__(self, json_path, tokenizer, sent2idx, ent2idx, max_length=32):
#         with open(json_path, "r") as f:
#             self.data = json.load(f)
#         self.tokenizer = tokenizer
#         self.sent2idx = sent2idx
#         self.ent2idx = ent2idx
#         self.max_length = max_length

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         entry = self.data[idx]
#         sentence = entry["sentence"]
#         # sentence classification label
#         cls_label = self.sent2idx[entry["task_A"]]

#         # tokenize with offsets so we can align NER spans
#         enc = self.tokenizer(
#             sentence,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors=None,
#             return_offsets_mapping=True,
#         )
#         input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
#         attention_mask = torch.tensor(enc["attention_mask"], dtype=torch.long)
#         offsets = enc["offset_mapping"]

#         # build token-level NER labels (default to “O”)
#         ner_labels = torch.zeros(self.max_length, dtype=torch.long)
#         for ent in entry["task_B"]:
#             start_char, end_char, typ = ent["start"], ent["end"], ent["type"]
#             for i, (o_start, o_end) in enumerate(offsets):
#                 if o_start == start_char:
#                     ner_labels[i] = self.ent2idx[f"B-{typ}"]
#                 elif o_start >= start_char and o_end <= end_char:
#                     ner_labels[i] = self.ent2idx[f"I-{typ}"]

#         return input_ids, attention_mask, torch.tensor(cls_label), ner_labels


# def collate_fn(batch):
#     input_ids, attention_mask, cls_labels, ner_labels = zip(*batch)
#     return {
#         'sentences': 
#         "input_ids":      torch.stack(input_ids),
#         "attention_mask": torch.stack(attention_mask),
#         "cls_labels":     torch.stack(cls_labels),
#         "ner_labels":     torch.stack(ner_labels),
#     }
