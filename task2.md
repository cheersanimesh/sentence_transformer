
## Task 2: Multi-Task Learning Expansion

### 2.1 Dummy Dataset  
We created a tiny toy dataset of 6 examples in `data/dummy_data.json`. Each entry has:  
- A raw sentence  
- A sentence‐level sentiment label (one of four classes)  
- A token‐level entity tag sequence (one of nine tags)  

Example entry:
```json
{
		"sentence": "Fetch Rewards is awesome and is America's number one app",
		"task_A": "technology",
		"task_B": [
		  { "text": "Fetch Rewards", "type": "ORG", "start": 0, "end": 13 },
		  { "text": "America", "type": "LOC", "start": 32, "end": 39 }
		]
}
```

### 2.2 Task Definitions  
1. **Sentence Classification (Sentiment)**  
   - **Labels (4):** `travel`, `technology`, `politics`, `other`  
   - **Output dim:** 4  

2. **Named Entity Recognition (NER)**  
   - **Tags (9):**  
     ```
     0: O
     1: B-PERSON
     2: I-PERSON
     3: B-LOC
     4: I-LOC
     5: B-ORG
     6: I-ORG
     7: B-DATE
     8: I-DATE
     ```  
   - **Output dim:** 9  

Both tasks share the same GPT-Neo encoder (from Task 1) but use separate projection heads.

### 2.3 Model Architecture  
![Multi-Task Architecture](images/task2_architecture.png)  

- **Shared Encoder**  
  - `GPT-Neo 125M` (hidden_dim = 768)  

- **Sentence Classification Head**  
  ```python
  nn.Sequential(
      nn.Dropout(0.1),
      nn.Linear(hidden_dim, 4)
  )
  ```  
  - Applies to the pooled sentence embedding → 4-way sentiment logits.

- **NER Head**  
  ```python
  nn.Sequential(
      nn.Dropout(0.1),
      nn.Linear(hidden_dim, 9)
  )
  ```  
  - Applies token-wise to each of the `seq_len` hidden states → (batch, seq_len, 9) tag logits.

### 2.4 Initial Outputs & Next Steps  
On our untrained projection heads, a forward pass over the 6 dummy sentences yields tensor shapes:  
`class_logits.shape == (6, 4)`  
`ner_logits.shape  == (6, seq_len, 9)`  
### Sample Output

Sentence: The Yellowstone National Park sees over four million visitors.

- **[Task A] Predicted category:** technology  
- **[Task B] Predicted entities:**
  - **Yellowstone National Park** (LOC) — spans characters 4–31  



Although the predictions are random initially, this confirms our model produces correctly-shaped outputs for both tasks. 
> With training and fine-tuning, the model will jointly learn representations that serve both sentence-level and token-level objectives, improving generalization and parameter efficiency.  
<!-- ```

### 2.4 Forward Pass (Pseudo-Code)  
```python
# 1. Encode
embeddings, token_embeddings = encoder(sentences)
#   embeddings: (batch, hidden_dim)
#   token_embeddings: (batch, seq_len, hidden_dim)

# 2. Sentence classification
class_logits = classification_head(embeddings)       # → (batch, 4)

# 3. Token-level NER
ner_logits = ner_head(token_embeddings)              # → (batch, seq_len, 9)
```

### 2.5 Losses & Multi-Task Objective  
- **Sentence classification loss**  
  ```python
  loss_class = CrossEntropyLoss(class_logits, sentiment_labels)
  ```  
- **NER loss** (ignore padding tokens)  
  ```python
  loss_ner = CrossEntropyLoss(
      ner_logits.view(-1, 9),
      entity_labels.view(-1),
      ignore_index=pad_token_id
  )
  ```  
- **Total loss**  
  ```
  L_total = α * loss_class + β * loss_ner
  ```  
  - Start with α = β = 1.0; adjust if one task’s loss dominates. -->