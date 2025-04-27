from datasets import load_dataset

def get_dataset(dataset_a ='ag_news', dataset_b='glue', split ='test'):
    ds_A = load_dataset(dataset_a, split=split)    # Task A: 4 classes
    ds_B = load_dataset(dataset_b, "sst2", split='split')  # Task B: binary sentiment


def prep_A(ex):
    return {"sentence": ex["text"], "label_A": ex["label"]}
def prep_B(ex):
    return {"sentence": ex["sentence"], "label_B": ex["label"]}

ds_A = ds_A.map(prep_A, remove_columns=ds_A.column_names)
ds_B = ds_B.map(prep_B, remove_columns=ds_B.column_names)

def collate_A(batch):
    sentences = [x["sentence"] for x in batch]
    labels_A  = torch.tensor([x["label_A"] for x in batch], dtype=torch.long)
    return sentences, labels_A

def collate_B(batch):
    sentences = [x["sentence"] for x in batch]
    labels_B  = torch.tensor([x["label_B"] for x in batch], dtype=torch.long)
    return sentences, labels_B


from torch.utils.data import DataLoader


loader_A = DataLoader(ds_A, batch_size=1, shuffle=True,  collate_fn=collate_A)
loader_B = DataLoader(ds_B, batch_size=1, shuffle=True,  collate_fn=collate_B)


# for (sents_A, labels_A), (sents_B, labels_B) in zip(loader_A, loader_B):
#     logits_A, _   = model(sents_A)
#     #loss_A        = criterionA(logits_A.to(device), labels_A.to(device))
#     # Task B forward + loss
#     _, logits_B   = model(sents_B)
#     #loss_B        = criterionB(logits_B.to(device), labels_B.to(device))

#     break
