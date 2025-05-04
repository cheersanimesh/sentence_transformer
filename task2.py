import torch
import torch.nn as nn                    
from models.sentence_transformer import SentenceTransformer
from models.multi_task_transformer import MultiTaskSentenceTransformer

import argparse
import os
import warnings
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid overly verbose tokenizer logs
warnings.filterwarnings("ignore")               # Silence all warnings


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#------------------------------------------------------------------------------#
#  Load Dummy Data
#------------------------------------------------------------------------------#
with open("data/dummy_data.json", "r") as f:
    data = json.load(f)

test_sentences = [entry["sentence"] for entry in data]

#------------------------------------------------------------------------------#
#  Task‑specific label vocabularies
#------------------------------------------------------------------------------#
sentence_class_labels = ["travel", "technology", "politics", "other"]

ner_label_map = {
    0: "O",
    1: "B-PERSON",  2: "I-PERSON",
    3: "B-LOC",    4: "I-LOC",
    5: "B-ORG",    6: "I-ORG",
    7: "B-DATE",   8: "I-DATE",
    9: "I-MISC",  10: "B-MISC"
}

#==============================================================================#
#  Main driver
#==============================================================================#
def main(args):

    #-----------------------------#
    # 1) Build / load the encoder #
    #-----------------------------#
    encoder_model = SentenceTransformer(
        backbone_type=args.encoder_backbone_type,     # e.g. "EleutherAI/gpt-neo-125M"
        pooling=args.encoder_backbone_pooling         # usually "mean" or "cls"
    )

    if args.encoder_model_path is not None:           # optional fine‑tuned checkpoint
        encoder_model = torch.load(args.encoder_model_path)

    #---------------------------------------------#
    # 2) Wrap encoder in a multi‑task transformer #
    #---------------------------------------------#
    multi_task_transformer_model = MultiTaskSentenceTransformer(
        encoder=encoder_model,
        pooling=args.model_pooling,
        num_classes_sentence_classification=args.num_classes_task_sentence_classification,
        num_ner_labels=args.num_classes_ner
    ).to(device)

    # Optionally load a *joint* model snapshot (encoder + heads)
    if args.model_load_path is not None:
        snapshot = torch.load(args.model_load_path)
        multi_task_transformer_model.load_state_dict(snapshot)

    #---------------------------#
    # 3) Run inference pipeline #
    #---------------------------#
    multi_task_transformer_model.eval()               # switch to eval mode
    tokenizer = multi_task_transformer_model.encoder.tokenizer

    with torch.no_grad():                            # inference only – disable grads
        cls_logits, ner_logits = multi_task_transformer_model(test_sentences)
        preds_cls = torch.argmax(cls_logits, dim=-1).tolist()
        preds_ner = torch.argmax(ner_logits, dim=-1).tolist()

    
    for sid, sentence in enumerate(test_sentences):
        print(f"\nSentence {sid}: «{sentence}»")
        print("-" * 60)

        # -------- Task A: sentence‑level class --------
        cls_idx = preds_cls[sid]
        print("[Task A] Predicted category:", sentence_class_labels[cls_idx])

        # -------- Task B: token‑level NER -------------
        print("\n[Task B] Named Entity Recognition (NER)\n")

        # Tokenize with offset and word‑id mapping (BERT‑style)
        enc        = tokenizer(sentence, return_offsets_mapping=True)
        toks       = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        offsets    = enc["offset_mapping"]           # [(char_start, char_end), ...]
        word_ids   = enc.word_ids()                  # map sub‑token → word index

        # Iterate over *distinct* words (skip [CLS]/[SEP] etc.)
        seen_words = set()
        for tok_idx, wid in enumerate(word_ids):
            if wid is None or wid in seen_words:
                continue
            seen_words.add(wid)

            # All sub‑token indices belonging to this word
            piece_idxs = [i for i, w in enumerate(word_ids) if w == wid]

            # Char span covering all pieces
            start = offsets[piece_idxs[0]][0]
            end   = offsets[piece_idxs[-1]][1]

            # Reconstruct the original word string
            word = tokenizer.convert_tokens_to_string([toks[i] for i in piece_idxs])

            # BIO tag – first piece is standard for beginning marker
            ent_label = ner_label_map[preds_ner[sid][piece_idxs[0]]]

            print(f"{word:<20s} [{start:>2d}:{end:<2d}]  →  {ent_label}")

#==============================================================================#
#  Script entry‑point – parse CLI and launch main()
#==============================================================================#
if __name__ == '__main__':

    # Ensure output dirs exist
    os.makedirs('images',      exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    parser = argparse.ArgumentParser()

    # --- Checkpoints ---
    parser.add_argument('--model_load_path',   type=str, default=None,
                        help='Path to a full multi‑task checkpoint (.pt)')
    parser.add_argument('--encoder_model_path', type=str, default=None,
                        help='Path to a standalone encoder checkpoint (.pt)')

    # --- Encoder backbone + pooling strategy ---
    parser.add_argument('--encoder_backbone_type',    type=str,
                        default='FacebookAI/roberta-base')
    parser.add_argument('--encoder_backbone_pooling', type=str, default='mean')
    parser.add_argument('--model_pooling',            type=str, default='mean',
                        help='Pooling used in the multi‑task heads')
    parser.add_argument('--pooling_op',               type=str, default='mean',
                        help='[Deprecated] kept for back‑compat')

    # --- Task‑head dimensionalities ---
    parser.add_argument('--num_classes_task_sentence_classification', type=int, default=4)
    parser.add_argument('--num_classes_ner',                          type=int, default=11)

    args = parser.parse_args()
    main(args)
