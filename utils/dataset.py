import json
import torch
from torch.utils.data import Dataset

class DummyMTLDataset(Dataset):
    """Reads the JSON list structure and emits tokenised inputs + BIO labels."""

    def __init__(self, json_path,sent2idx, ent2idx, tokenizer):
        self.tokenizer = tokenizer
        with open(json_path, "r") as f:
            self.data = json.load(f)
        
        self.sent2idx = sent2idx
        self.ent2idx = ent2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sent = item["sentence"]
        cls_label = item['task_A']
        ents = item["task_B"]

        enc = self.tokenizer(
            sent,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        labels = [self.ent2idx["O"]] * len(enc.offset_mapping)

        # annotate tokens
        for ent in ents:
            e_start, e_end = ent["start"], ent["end"]
            e_type = ent["type"].upper()
            b_tag = f"B-{e_type}" if f"B-{e_type}" in self.ent2idx else "B-MISC"
            i_tag = f"I-{e_type}" if f"I-{e_type}" in self.ent2idx else "I-MISC"

            for tidx, (t_start, t_end) in enumerate(enc.offset_mapping):
                # skip special tokens (t_start==t_end==0 for GPT‑2)
                if t_start == t_end:
                    continue
                if t_start >= e_end:
                    break  # past the entity span
                if t_end <= e_start:
                    continue  # before the entity span

                # overlap exists → assign B‑ or I‑ label
                labels[tidx] = self.ent2idx[b_tag] if t_start == e_start else self.ent2idx[i_tag]

        enc.pop("offset_mapping")
        enc["labels"] = labels
        return {
            "sentence":sent,
            "cls_label": self.sent2idx[cls_label],
            "spans": enc['labels']
        }
        #return {k: torch.tensor(v) for k, v in enc.items()}

def collate_fn(batch):
    sentences   = [b["sentence"]   for b in batch]
    cls_labels  = torch.tensor([b["cls_label"] for b in batch], dtype=torch.long)
    spans_batch = [b["spans"]      for b in batch]
    return {
        "sentences":  sentences,
        "cls_labels": cls_labels,
        "spans":      spans_batch
    }