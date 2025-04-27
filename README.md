# sentence_transformer


## Task -1


For the sentence Transformer the model , the overall sentence transformer looks like as following:

\\ add image

It consists of the following components :

1. A BPE based GPT tokenizer
2. A pretrained GPTNEOModel as the transformer backbone. The core advantage behind using the GPTNEOModel is because of :
	
	a. It uses the **Rotary Positional Embeddings** as the embeddings layer. This embedding layer brings an added advantage of understanding of relative positions amongst the tokens.
	
	b. The  <Add more good things about the model>
3. An added pooling layer >


We tested the model on the following test sentences and have used **PCA** and **t-SNE** for visualisation as depicted in the table :

Visualisation using PCA            |  Visualisation using t-SNE
:-------------------------:|:-------------------------:
![](images/part_1_pca_visualisation.jpg)  |  ![](images/part_1_tsne_visualisation.jpg)



We can observe the words having similar semantic meanings and features are closer in the plots as compared to other sentences.


## Task -2

