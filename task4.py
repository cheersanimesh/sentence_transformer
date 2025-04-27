import torch
import torch.nn as nn
from models.sentence_transformer import SentenceTransformer
from models.multi_task_transformer import MultiTaskSentenceTransformer
import argparse
import os
import warnings

def main():
	pass

if __name__=='__main__':

	os.makedirs('images', exist_ok = True)
	os.makedirs('checkpoints', exist_ok = True)

	parser = argparse.ArgumentParser()

	parser.add_argument('--model_load_path', type = str, default=None)
	parser.add_argument('--')	

	args = parser.parse_args()
	main(args)