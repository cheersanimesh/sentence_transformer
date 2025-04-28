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