## Task -3 Training Considerations

### 3.1 Freezing the Entire Network
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
- **Transfer learning:** follow a staged unfreezing protocol with cautious learning rates and early stopping to balance adaptation vs. forgetting.