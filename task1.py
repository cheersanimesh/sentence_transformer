import torch
import torch.nn as nn
from models.sentence_transformer import SentenceTransformer
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.visualise import plot_projection
import os
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress all warnings
warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Assuming single gpu infra

test_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "I love machine learning!",
    "I love doughnuts",
    "I love bread",
    "Transformers are amazing.",
    "This is a test sentence."
]

def main(args):
	
	model = SentenceTransformer(
		backbone_type = args.backbone,
		pooling = args.pooling_op,
	).to(device)

	with torch.no_grad():
		sentence_embeddings , _ = model(test_sentences)
	
	sentence_embeddings = sentence_embeddings.cpu().numpy()

	## visualise using PCA 

	pca = PCA(n_components=2) 
	proj_pca = pca.fit_transform(sentence_embeddings)

	plot_projection(proj_pca, "PCA of Sentence Embeddings",sentences = test_sentences,  file_dest='images/part_1_pca_visualisation.jpg')

	## visualise using TSNE

	tsne = TSNE( n_components=2, init="pca", random_state=42, perplexity=2)
	proj_tsne = tsne.fit_transform(sentence_embeddings)

	plot_projection(proj_tsne, "t-SNE of Sentence Embeddings",sentences = test_sentences,  file_dest='images/part_1_tsne_visualisation.jpg')

	## saving the model
	if args.save_model:
		torch.save(model, args.model_write_loc)


if __name__=='__main__':

	os.makedirs('images', exist_ok = True)
	os.makedirs('checkpoints', exist_ok = True)
	
	parser = argparse.ArgumentParser()

	parser.add_argument("--backbone",type = str, default= 'EleutherAI/gpt-neo-125M')
	parser.add_argument("--pooling_op",type=str, default= 'mean' )
	parser.add_argument("--save_model", type = bool, default = True)
	parser.add_argument("--model_write_loc", type=str, default= 'checkpoints/sentence_transformer.pt')

	args = parser.parse_args()
	main(args)