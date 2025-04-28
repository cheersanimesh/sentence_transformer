# Sentence Transformer

## Installation 

## Guide to Codebase

The results and inferences are compiled in task1.md 

<!-- ## Task -1


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



We can observe the words having similar semantic meanings and features are closer in the plots as compared to other sentences. -->


## Task -2

The results and discussions are compiled in task2.md

<!-- For this part and we have created a dummy dataset consisting of 5 sentences locatated in 'data/dummy_data.json'

Also for this part, we have chosen the following tasks for Multi-Task Learning Framework

1. Sentence classification for sentiment (Output Dim: **4**) , it could belong to one of the four labels : [["travel", "technology", "politics", "other"] 
2. Named entity Recognition at token level for the sentences (Output Dim : **9**) i.e each token could fall into : 0: "O",
    1: "B-PERSON",
    2: "I-PERSON",
    3: "B-LOC",
    4: "I-LOC",
    5: "B-ORG",
    6: "I-ORG",
    7: "B-DATE",
    8: "I-DATE",

Both the task share the same enocder however have different set of projection heads. This design was adopted to keep the model and to rely on the GPTNeoModel due to reasons mentioned in the part 1.

The overall model looks in the following way :

\image

We passed the sentences to the model to get the following output :


Since our projection head is untrained and randomly initialized, the model's output is not what we would expect. However we can see that the model, has correct output format and on further finetuning the model, one could expect it to perform better on the two tasks. -->






## Task -3 Training Considerations

The results and discussions are compiled in task3.md
<!-- ### 3.1 Freezing the Entire Network
- **Definition**  
  No parameters—neither backbone nor task heads—are updated during training.
- **Use case**  
  - Pure feature extraction: use the pretrained model “as is” and train external classifiers.  
  - Quick baseline to gauge out-of-the-box performance.
- **Pros**  
  - Zero risk of overfitting on small task data.  
  - Very fast training (no backprop).
- **Cons**  
  - No domain or label-space adaptation.  
  - Cannot train any heads internally if they are frozen too.

---

### 3.2 Freezing Only the Transformer Backbone
- **Definition**  
  Keep `encoder.backbone` frozen, but train both `classifier_A` and `ner_head`.
- **Use case**  
  - Linear probing: evaluate how well fixed pretrained embeddings support your tasks.  
  - Compute- or data-limited scenarios.
- **Pros**  
  - Efficient: only head parameters are updated.  
  - Stable: preserves pretrained representations.  
  - Easy to interpret: isolates the power of pretrained features.
- **Cons**  
  - If downstream tasks diverge from pretraining, fixed features may be suboptimal.  
  - Task heads alone may lack capacity to correct representation deficiencies.

---

### 3.3 Freezing Only One Task Head
- **Definition**  
  Freeze one head (e.g. `classifier_A`), while fine-tuning the backbone and the other head (`ner_head`), or vice versa.
- **Use case**  
  - Protecting a high-priority task from degradation when training on another.  
  - Sequential or continual learning workflows.
- **Pros**  
  - Selective plasticity: adapt to a new task without completely unlearning the frozen head.  
  - The frozen head regularizes shared representation learning.
- **Cons**  
  - Backbone updates may still hurt the frozen head’s performance if tasks conflict.  
  - Requires careful tuning (e.g., learning rates, gradient projection) to avoid negative transfer.

---

### 3.4 Transfer-Learning Strategy
1. **Choice of Pre-trained Model**  
   - _Lightweight iteration:_ `EleutherAI/gpt-neo-125M` (fast to train).  
   - _Higher capacity:_ consider larger GPT-Neo (1.3B) or “all-MiniLM” models for stronger embeddings.
2. **Layers to Freeze & Unfreeze**  
   1. **Stage 1 (Linear Probe):** Freeze all transformer layers; train only the heads.  
   2. **Stage 2:** Unfreeze the top 1–2 transformer blocks; fine-tune backbone + heads at a low learning rate (e.g. 1e-5).  
   3. **Stage 3 (Optional):** Gradually unfreeze lower layers (“gradual unfreezing”) while monitoring dev-set performance.
3. **Rationale**  
   - **Lower layers** capture general syntax/semantics—keep them frozen to retain robustness.  
   - **Higher layers** encode task-specific patterns—unfreeze them to adapt to your classification & NER distributions.  
   - Use **discriminative learning rates** (higher for heads, lower for backbone) to speed up head training and protect pretrained weights.

---

### Summary of Recommendations
- **Freeze everything** when you need raw embeddings or a baseline evaluation.  
- **Freeze only the backbone** for a fast linear-probe setup.  
- **Freeze one head** to safeguard an existing task while adapting to another.  
- **Transfer learning:** follow a staged unfreezing protocol with cautious learning rates and early stopping to balance adaptation vs. forgetting. -->


## Task - 4

For Task -4 our goal is to train the Multi-Task-Sentence-Transformer network for multi task. Here we are performing rating jointly training the projection heads with a low learning rate. We choose to do the following: <add reason>


