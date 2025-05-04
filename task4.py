import torch
import torch.nn as nn
from models.sentence_transformer import SentenceTransformer
from models.multi_task_transformer import MultiTaskSentenceTransformer
import argparse
import os
import warnings
from utils.dataset import DummyMTLDataset, collate_fn
from torch.utils.data import DataLoader
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress all warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Assuming single gpu infra

def main(args):

    encoder = SentenceTransformer(
        backbone_type=args.encoder_type, 
        pooling=args.pooling
    )
    model = MultiTaskSentenceTransformer(
        encoder,
        hidden_size=args.hidden_size,
        num_classes_sentence_classification=len(args.sentence_class_labels),
        num_ner_labels=len(args.ner_label_map),
        pooling=args.pooling
    ).to(device)

    if args.model_load_path is not None:
        model = torch.load(args.model_load_path)
    

    # Freezing the encocder weights
    if args.freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    if args.optimizer == 'adamW':
        optimizer  = torch.optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.sgd(model.parameters, lr = args.lr)
    else:
        raise NotImplementedError(f"Unknown optimizer {args.optimizer}")


    if args.criterion_sentence_classification == 'cross_entropy':
        criterion_sentence_classification =  nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    if args.criterion_ner == 'cross_entropy':
        criterion_ner = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    ## prepare dataset

    sent2idx = {lab: i for i, lab in enumerate(args.sentence_class_labels)}
    ent2idx = {lab: i for i, lab in enumerate(args.ner_label_map)}
    idx2ent = {i:lab for i, lab in enumerate(args.ner_label_map)}
    tokenizer = model.encoder.tokenizer

    dataset = DummyMTLDataset(json_path = args.data_path , tokenizer = tokenizer, sent2idx = sent2idx, ent2idx = ent2idx)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    ## train loop

    num_epochs = args.epochs

    # 5) training loop
    model.train()
    for epoch in range(1, num_epochs+1):
        total_loss = 0.0
        for batch in loader:
            sentences   = batch["sentences"]
            cls_labels  = batch["cls_labels"].to(device)
            spans_batch = batch["spans"]
            # now a single forward call on raw strings:
            logits_cls, logits_ner = model(sentences)
            # sentence‐classification loss
            loss_sentence_classification = criterion_sentence_classification(logits_cls, cls_labels)

           
            B, T, L = logits_ner.shape
            target_ner = torch.zeros(B, T, dtype=torch.long, device=device)

            # for i, spans in enumerate(spans_batch):
            #     for ent in spans:
            #         start, end, typ = ent["start"], ent["end"], ent["type"]

            loss_ner = criterion_ner(
                logits_ner.view(-1, L),
                target_ner.view(-1)
            )
            # import ipdb; ipdb.set_trace()
            loss = loss_sentence_classification + loss_ner
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}/{args.epochs} — avg loss {total_loss/len(loader):.4f}")

    ## inference
    
    test_sentences = [
        "Fetch Rewards is awesome and is America number one app",
        'Hello Fetch Rewards'
    ]

    for sent_idx, sentence in enumerate(test_sentences):

        encoding = tokenizer(sentence, return_offsets_mapping=True)
        input_ids = encoding["input_ids"]
        toks = tokenizer.convert_ids_to_tokens(input_ids)
        logits_cls, logits_ner = model(sentence)

        word_ids = encoding.word_ids()

        preds_cls , preds_ner= logits_cls.argmax(dim=-1), logits_ner.argmax(dim=-1)[0]
        
        word_tokens = defaultdict(list)
        for idx, tok in enumerate(toks):
            wid = word_ids[idx]
            if wid is None:
                continue
            word_tokens[wid].append(tok)

        
        words_map = {
            wid: tokenizer.convert_tokens_to_string(tok_list)
            for wid, tok_list in word_tokens.items()
        }

        print(f"Sentence id : {sent_idx} <<-->> {sentence} \n")
        
        for idx, pred in enumerate(preds_ner[1: len(toks)+1]):
            token_idx = idx + 1           # because we skipped [CLS] at position 0
            wid       = word_ids[token_idx]
            if wid is None:
                continue
            ent_label = idx2ent[pred.item()]
            word      = words_map[wid]
            print(f"{word} --> {ent_label}")
        
    

    # 5) save
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    torch.save(model.state_dict(), args.model_save_path)
    print("Saved model to", args.model_save_path)

	
	
	



if __name__=='__main__':

    os.makedirs('images', exist_ok = True)
    os.makedirs('checkpoints', exist_ok = True)
    os.makedirs('data', exist_ok = True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--freeze_encoder', type = bool, default = False)
    parser.add_argument('--data_path', type=str, default='data/dummy_data.json')
    parser.add_argument('--model_load_path', type = str, default=None)
    parser.add_argument('--encoder_type', type = str, default = 'EleutherAI/gpt-neo-125M')
    parser.add_argument('--pooling', type = str, default = 'mean')
    parser.add_argument('--hidden_size',type= int, default = 768)

    

    parser.add_argument('--epochs', type = int, default= 10)
    parser.add_argument('--lr', type = float, default = 5e-3)
    parser.add_argument('--optimizer', type = str, default = 'adamW')	
    parser.add_argument('--batch_size', type = int, default = 32)
    parser.add_argument(
        '--sentence_class_labels',
        nargs='+',
        default=['travel', 'technology', 'politics', 'entertainment']
    )
    parser.add_argument(
        '--ner_label_map',
        nargs='+',
        default=[
            'O',
            'B-PERSON','I-PERSON',
            'B-LOC','I-LOC',
            'B-ORG','I-ORG',
            'B-DATE','I-DATE','I-MISC', 'B-MISC'
        ]
    )


    parser.add_argument('--criterion_sentence_classification', type = str, default = 'cross_entropy')
    parser.add_argument('--criterion_ner', type = str, default = 'cross_entropy')

    parser.add_argument('--max_length', type = int, default= 32)
    parser.add_argument('--model_save_path', type = str, default = 'checkpoints/task4_model.pt')

    args = parser.parse_args()
    main(args)

