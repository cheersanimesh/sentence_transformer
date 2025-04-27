import torch
import torch.nn as nn
from models.sentence_transformer import SentenceTransformer_Pretrained_Backbone
import argparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils.visualise import plot_projection


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
	
	model = SentenceTransformer_Pretrained_Backbone(
		backbone_type = args.backbone,
		pooling = args.pooling,
	)

	with torch.no_grad():
		sentence_embeddings = model(sentences)
	
	sentence_embeddings = sentence_embeddings.cpu().numpy()

	## visualise using PCA 

	pca = PCA(n_components=2) 
	proj_pca = pca.fit_transform(embeddings)

	plot_projection(proj_pca, "PCA of Sentence Embeddings", file_dest='images/part_1_pca_visualisation.jpg')

	## visualise using TSNE

	tsne = TSNE( n_components=2, init="pca", random_state=42, perplexity=2)
	proj_tsne = tsne.fit_transform(embeddings)

	plot_projection(proj_tsne, "t-SNE of Sentence Embeddings", file_dest='images/part_1_tsne_visualisation.jpg')




if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("backbone",type = str, default= 'EleutherAI/gpt-neo-125M')
	parser.add_argument("pooling_op",type=str, default= 'mean' )
	# parser.add_argument("--age", type=int, help="Your age")
	# parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
	args = parser.parse_args()
	main(args)