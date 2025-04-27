import torch
import torch.nn as nn
from models.sentence_transformer import SentenceTransformer
from models.multi_task_transformer import MultiTaskSentenceTransformer
import argparse
import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress all warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Assuming single gpu setup

test_sentences = [
	'hello world how are you', 
	'i am fine so let us go on a beach',
]

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

	with torch.no_grad():
		classification_output, ner_task_output  = multi_task_transformer_model(test_sentences)

	##TODO: analyse the outputs

	
	





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