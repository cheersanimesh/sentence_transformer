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
}

def main(args):
	
	encoder_model = SentenceTransformer(
		backbone_type = args.encoder_backbone_type,
		pooling = args.encoder_backbone_pooling
	)

	if args.encoder_model_path is not None:
		## load model
		encoder_model = torch.load(args.encoder_model_path)
	

	multi_task_transformer_model = MultiTaskSentenceTransformer(
		encoder = encoder_model,
		pooling = args.model_pooling,
		num_classes_sentence_classification = args.num_classes_task_sentence_classification,
		num_ner_labels = args.num_classes_ner
	).to(device)

	## Testing the inference of the model

	multi_task_transformer_model = multi_task_transformer_model.eval()
	encoder = multi_task_transformer_model.encoder
	tokenizer = encoder.tokenizer

	with torch.no_grad():
		logits_sentence_classification, ner_logits = multi_task_transformer_model(test_sentences)
		preds_cls = torch.argmax(logits_sentence_classification, dim=-1).tolist()
		preds_ner = torch.argmax(ner_logits, dim = -1).tolist()

	for i, sent in enumerate(test_sentences):
		print(f"\nSentence: “{sent}”")
			# --- Task A ---
		cls_idx = preds_cls[i]
		print("  [Task A] Predicted category:", sentence_class_labels[cls_idx])

		tokens = tokenizer.tokenize(sent)        # e.g. ["[CLS]", "Alice", "went", …, "[SEP]"]
		ner_ids = preds_ner[i]             # list of same length
		entities = []
		current_ent = None
		for tok, lid in zip(tokens, ner_ids):
			label = ner_label_map[lid]
			if label.startswith("B-"):
				if current_ent:
					entities.append(current_ent)
				current_ent = {"type": label[2:], "tokens": [tok]}
			elif label.startswith("I-") and current_ent:
				current_ent["tokens"].append(tok)
			else:
				if current_ent:
					entities.append(current_ent)
					current_ent = None
		if current_ent:
			entities.append(current_ent)

		# pretty‐print the spans
		print("  [Task B] Predicted entities:")
		for ent in entities:
			text = encoder.tokenizer.convert_tokens_to_string(ent["tokens"])
			print(f"    • {ent['type']}: “{text}”")




if __name__=='__main__':

	os.makedirs('images', exist_ok = True)
	os.makedirs('checkpoints', exist_ok = True)


	parser = argparse.ArgumentParser()
	parser.add_argument('--encoder_backbone_type', type = str, default = 'EleutherAI/gpt-neo-125M')
	parser.add_argument('--encoder_backbone_pooling', type = str, default = 'EleutherAI/gpt-neo-125M')
	parser.add_argument('--model_pooling', type = str, default='mean')
	parser.add_argument("--encoder_model_path",type = str, default= None)
	parser.add_argument("--pooling_op",type=str, default= 'mean' )
	
	parser.add_argument('--num_classes_task_sentence_classification', type = int, default = 4)
	parser.add_argument('--num_classes_ner', type = int, default = 9)
	
	args = parser.parse_args()
	
	main(args)



### Quick ner inference 
'''

model.eval()
sent = "Barack Obama was born in Hawaii ."
tok = tokenizer(sent,
                return_tensors="pt",
                is_split_into_words=True,
                truncation=True,
                padding="longest")
with torch.no_grad():
    _, _, ner_logits = model(tok["input_ids"].to(device),
                             tok["attention_mask"].to(device))
# pick highest‐scoring label per token
pred_ids = ner_logits.argmax(-1)[0].cpu().tolist()
# map back to words
word_ids = tok.word_ids(batch_index=0)
prev_idx = None
entities = []
for idx, wid in enumerate(word_ids):
    if wid is None:
        continue
    if wid != prev_idx:
        tag = label_list[pred_ids[idx]]
        token = tok.tokens()[idx]
        entities.append((token, tag))
    prev_idx = wid

print(entities)

'''