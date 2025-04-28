## Task 1: Sentence Transformer Implementation

### Model Architecture  
![Sentence Transformer Architecture](images/task1_architecture.png)

Our sentence encoder consists of three main components:

1. **BPE-Based GPT Tokenizer**  
   - Uses Byte-Pair Encoding (BPE) to split text into subword units.  
   - **Why BPE?**  
     - Handles rare or out-of-vocabulary words by decomposing them into known subwords.  
     - Keeps vocabulary size manageable while minimizing “unknown” tokens.

2. **Pretrained GPT-Neo Backbone**  
   We selected the `GPT-Neo 125M` model for its:  
   - **Rotary Positional Embeddings (RoPE):**  
     - Encodes relative token positions, improving generalization to varied sequence lengths.  
   - **Dense Self-Attention:**  
     - Captures rich, pairwise contextual dependencies essential for nuanced sentence meaning.  
   - **Open-Source & Scalable Architecture:**  
     - Easily scaled to larger sizes (e.g., 1.3B parameters) without changing code.

3. **Mean-Pooling → Projection → Normalization**  
   - **Mean-Pooling:** Averages the final hidden states into one fixed-length vector.  
   - **Linear Projection:** Maps the pooled vector down to a 512-dimensional embedding.  
   - **ℓ₂-Normalization:** Ensures all embeddings lie on the unit hypersphere, making cosine similarity meaningful.

---

### Embedding Demonstration

**Sample Sentences**  
- “The cat sat on the mat.”  
- “A feline lounged on the carpet.”  
- “I enjoy reading books about history.”  
- “Studying past events fascinates me.”

We encoded these into 512-dim vectors and visualized them in 2D:

| **PCA Projection**                               | **t-SNE Projection**                             |
|:------------------------------------------------:|:------------------------------------------------:|
| ![](images/part_1_pca_visualisation.jpg)         | ![](images/part_1_tsne_visualisation.jpg)        |

**Observation:**  
Semantically similar sentences (e.g., “cat…mat” vs. “feline…carpet”) form tight clusters, confirming our model captures high-level meaning beyond surface token overlap.

---

### Design Choices & Rationale

- **Tokenizer:**  
  BPE balances vocabulary size and coverage, reducing unknown tokens.  
- **Backbone:**  
  GPT-Neo’s RoPE layer and self-attention suit sentence-level encoding; being open-source allows experimentation across sizes.  
- **Pooling + Projection:**  
  Mean-pooling is simple and effective; a projection layer controls embedding dimension; ℓ₂-normalization yields consistent similarity metrics.
