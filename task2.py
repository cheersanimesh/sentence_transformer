import torch
import torch.nn as nn
from models.sentence_transformer import SentenceTransformer
from models.multi_task_transformer import MultiTaskSentenceTransformer
import argparse
import os
import warnings
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress all warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Assuming single gpu setup


with open("data/dummy_data.json", "r") as f:
    data = json.load(f)

test_sentences = [entry["sentence"] for entry in data]

sentence_class_labels = ["travel", "technology", "politics", "other"] 

ner_label_map = {
    0: "O",
    1: "B-PERSON",
    2: "I-PERSON",
    3: "B-LOC",
    4: "I-LOC",
    5: "B-ORG",
    6: "I-ORG",
    7: "B-DATE",
    8: "I-DATE",
	9: "I-MISC",
	10:"B-MISC"
}


def main(args):

	# Initialize the shared sentence encoder
	encoder_model = SentenceTransformer(
		backbone_type = args.encoder_backbone_type,
		pooling = args.encoder_backbone_pooling
	)
	
	# If a pre-trained encoder checkpoint is provided, load it
	if args.encoder_model_path is not None:
		## load model
		encoder_model = torch.load(args.encoder_model_path)
	
	
	# Build the multi-task model using our encoder plus two heads
	multi_task_transformer_model = MultiTaskSentenceTransformer(
		encoder = encoder_model,
		pooling = args.model_pooling,
		num_classes_sentence_classification = args.num_classes_task_sentence_classification,
		num_ner_labels = args.num_classes_ner
	).to(device)
	
	if args.model_load_path is not None:
		snapshot = torch.load(args.model_load_path)
		multi_task_transformer_model.load_state_dict(snapshot)
		

	## Testing the inference of the model

	multi_task_transformer_model = multi_task_transformer_model.eval()
	encoder = multi_task_transformer_model.encoder
	tokenizer = encoder.tokenizer

	with torch.no_grad():
		logits_sentence_classification, ner_logits = multi_task_transformer_model(test_sentences)
		preds_cls = torch.argmax(logits_sentence_classification, dim=-1).tolist()
		preds_ner = torch.argmax(ner_logits, dim = -1).tolist()

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
			start = offsets[piece_idxs[0]][0] +1
			end   = offsets[piece_idxs[-1]][1]

			# Reconstruct the original word string
			word = tokenizer.convert_tokens_to_string([toks[i] for i in piece_idxs])

			# BIO tag – first piece is standard for beginning marker
			ent_label = ner_label_map[preds_ner[sid][piece_idxs[0]]]

			print(f"{word:<20s} [{start:>2d}:{end:<2d}]  →  {ent_label}")
	




if __name__=='__main__':

	os.makedirs('images', exist_ok = True)
	os.makedirs('checkpoints', exist_ok = True)


	parser = argparse.ArgumentParser()
	parser.add_argument('--model_load_path', type=str, default =None)

	parser.add_argument('--encoder_backbone_type', type = str, default = 'EleutherAI/gpt-neo-125M')
	parser.add_argument('--encoder_backbone_pooling', type = str, default = 'EleutherAI/gpt-neo-125M')
	parser.add_argument('--model_pooling', type = str, default='mean')
	parser.add_argument("--encoder_model_path",type = str, default= None)
	parser.add_argument("--pooling_op",type=str, default= 'mean' )
	

	parser.add_argument('--num_classes_task_sentence_classification', type = int, default = 4)
	parser.add_argument('--num_classes_ner', type = int, default = 11)
	
	args = parser.parse_args()
	
	main(args)