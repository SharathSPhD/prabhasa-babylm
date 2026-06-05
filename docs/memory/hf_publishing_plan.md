# PRABHĀSA HF Hub Publishing Plan

**Publication target:** `qbz506/prabhasa-b_ss-0.1` (Model) + `qbz506/prabhasa-corpus-from-grammar` (Dataset)

**Timeline:** Post-paper acceptance. Model and dataset repositories are prepared and staged for publication under Sharaths GitHub account.

---

## Part 1: Model Publication (`qbz506/prabhasa-b_ss-0.1`)

### 1.1 Pre-publication checklist

- [ ] Model checkpoint exported to HF format (safetensors + config.json + tokenizer)
  - Location: `/data/hf_export/arm_A_seed_0/` (or final chosen arm)
  - Files required:
    - `config.json` (ElcPsalmHFConfig with auto_map)
    - `model.safetensors` (weights)
    - `tokenizer.json` (HF tokenizer)
    - `tokenizer_config.json`
    - `modeling_elc_psalm.py` (ElcPsalmForMaskedLM, ElcPsalmModel)
    - `configuration_elc_psalm.py` (ElcPsalmHFConfig)

- [ ] Verify model loads correctly:
  ```python
  from transformers import AutoModelForMaskedLM, AutoTokenizer
  model = AutoModelForMaskedLM.from_pretrained(
      "qbz506/prabhasa-b_ss-0.1",
      trust_remote_code=True
  )
  tok = AutoTokenizer.from_pretrained("qbz506/prabhasa-b_ss-0.1")
  ```

- [ ] Verify demo notebooks run on Colab (CPU/GPU)
  - `demo_01_prabhasa_inference.ipynb`
  - `demo_02_mechanism_visualization.ipynb`
  - `demo_03_reproduce_blimp.ipynb`

- [ ] Create model card (README.md)

- [ ] Sign-off from project team

### 1.2 Model card template (README.md)

```markdown
---
language:
  - en
  - sa
license: apache-2.0
library_name: transformers
tags:
  - masked-lm
  - bert
  - structural-generalization
  - compositional-reasoning
  - morphology
---

# PRABHĀSA-B (Strict-Small)

**PRABHĀSA:** Pāṇinian Structured pretraining for Small Language Models.

## Model Details

**Architecture:** ELC (Encoder-Language-Core), BERT-family MLM encoder
- **Parameters:** 114M
- **Vocabulary:** 20,000 (SentencePiece BPE)
- **Layers:** 14 transformer blocks
- **Attention heads:** 12
- **Hidden dim:** 768
- **Max sequence length:** 128

**Pretraining:**
- **Data:** BabyLM Strict-Small (10M tokens, 4K types)
- **Objective:** Hybrid MLM (30% masking) + CLM (50% weight each)
- **Mechanisms:**
  - **Vidyut N-hot morpheme-boundary embeddings:** Assign morphological roles to tokens (word-initial, continuation, prefix, suffix, Sanskrit morphology)
  - **Paribhāṣā kāraka-aware adaptive masking:** Adjust token masking probability based on thematic role and morpheme boundaries

**Evaluation:**
- Zero-shot structured generalization: SCAN, COGS, CFQ, BLiMP, EWoK
- BLiMP score: [RESULT] (12 phenomena, ~67K minimal pairs)
- Grammatical agreement, long-range dependencies, movement, binding, entity tracking

## Usage

### Masked LM (MLM) — Fill-in-the-blank

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

model = AutoModelForMaskedLM.from_pretrained(
    "qbz506/prabhasa-b_ss-0.1",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("qbz506/prabhasa-b_ss-0.1")

text = "The cat [MASK] on the mat."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

# Top-k predictions at mask position
mask_pos = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
probs = torch.softmax(logits[0, mask_pos], dim=-1)
top_k = torch.topk(probs, 5)
for prob, idx in zip(top_k.values, top_k.indices):
    print(f"{tokenizer.decode([idx]):<10} {prob.item():.4f}")
```

### Pseudo-log-likelihood (PLL) — BLiMP-style scoring

```python
@torch.no_grad()
def score_sentence(text: str) -> float:
    ids = tokenizer(text, return_tensors="pt").input_ids
    total = 0.0
    for i in range(1, ids.size(1) - 1):
        masked = ids.clone()
        gold = ids[0, i].item()
        masked[0, i] = tokenizer.mask_token_id
        logits = model(masked).logits[0, i]
        total += torch.log_softmax(logits, dim=-1)[gold].item()
    return total

grammatical = "The cat sleeps on the mat."
ungrammatical = "The cat sleep on the mat."

score_good = score_sentence(grammatical)
score_bad = score_sentence(ungrammatical)
print(f"Grammatical: {score_good:.2f}, Ungrammatical: {score_bad:.2f}")
assert score_good > score_bad, "Model prefers grammatical sentence"
```

### Fine-tuning on downstream tasks

```python
from transformers import AutoModel
import torch.nn as nn

base_model = AutoModel.from_pretrained(
    "qbz506/prabhasa-b_ss-0.1",
    trust_remote_code=True
)

# Attach your task head
class ClassifierHead(nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, num_labels)
    
    def forward(self, hidden_state):
        x = self.dense(hidden_state[:, 0])  # [CLS] pooling
        return self.out(x)

head = ClassifierHead(768, num_labels=2)
# Training: standard Hugging Face Trainer or torch training loop
```

## Training Details

- **Optimizer:** AdamW (lr=1e-4, β₁=0.9, β₂=0.999)
- **Batch size:** 256 (8 GPU × 32 per GPU)
- **Gradient accumulation:** 1 step
- **Warmup:** 1000 steps
- **Total steps:** ~40K (10M tokens ÷ 256)
- **Weight decay:** 0.01
- **Clip norm:** 1.0

## Mechanism Details

### Vidyut N-hot Morpheme Embeddings

Each token is assigned a 10-dimensional binary vector encoding morphological roles:

1. `bpe_word_start` — SentencePiece word-initial token (begins with ▁)
2. `bpe_continuation` — continuation piece (no ▁ prefix)
3. `bpe_single` — single-token word
4. `bpe_suffix_like` — ends with -ing, -ed, -er, -ly, -tion, -ness, etc.
5. `bpe_prefix_like` — starts with un-, re-, pre-, dis-, etc.
6. `vidyut_root` — Sanskrit dhātu (root)
7. `vidyut_pratyaya` — Sanskrit affix/suffix
8. `vidyut_sandhi` — Sanskrit sandhi junction
9. `vidyut_krit` — Sanskrit krit pratyaya (verbal derivative)
10. `vidyut_taddhita` — Sanskrit taddhita pratyaya (nominal derivative)

The N-hot vector is projected to d_model and added to the token embedding. This is a lightweight, interpretable extension that grounds the model in linguistic structure.

### Paribhāṣā Kāraka-aware Masking

Token masking probability during pretraining adapts based on morpheme role and estimated thematic role (kāraka):

- **Role-bearing tokens** (word-initial heads, main verbs, agent nouns): 35% masking
- **Continuation pieces:** 25% masking
- **Bound morphemes** (prefixes, suffixes): 20% masking
- **Baseline (BERT):** 15% (static, uniform)

This increases sample efficiency by encouraging the model to learn role-sensitive, compositional representations early in training.

## Results

### Zero-shot BLiMP (minimal-pair scoring, PLL method)

| Phenomenon | Accuracy |
|---|---|
| Agreement violations | [X]% |
| Anaphor agreement | [X]% |
| Binding | [X]% |
| Control/Raising | [X]% |
| Determiner-Noun agreement | [X]% |
| Ellipsis | [X]% |
| Filler-Gap dependencies | [X]% |
| Irregular forms | [X]% |
| Island effects | [X]% |
| NPI licensing | [X]% |
| Quantifiers | [X]% |
| Subject-Verb agreement | [X]% |
| **Mean** | **[X]%** |

### Other benchmarks (zero-shot)

- **SCAN (compositional generalization):** [X]%
- **COGS:** [X]%
- **CFQ:** [X]%
- **EWoK (entity/world knowledge transfer):** [X]%

*Fill in [X] with published results from the paper.*

## Citation

If you use PRABHĀSA in published research, please cite:

```bibtex
@article{sharaths2025prabhasa,
  title={PRABHĀSA: Pāṇinian Structured Pretraining for Small Language Models},
  author={Sharaths, [Affiliation]},
  journal={[Journal/Conference]},
  year={2025}
}
```

## License

Apache 2.0

## Disclaimer

PRABHĀSA is designed for **structural generalization and compositional reasoning**, not frontier language understanding. It should not be used for:
- Applications requiring world knowledge (facts, dates, names)
- High-stakes decision-making
- Generation of factual claims without verification

The model excels at:
- Linguistic agreement and binding phenomena
- Compositional generalization (SCAN, COGS, CFQ)
- Morphology-aware representations
- Sample-efficient pretraining on structured data

## Demo Notebooks

- **[demo_01_prabhasa_inference.ipynb](demo_01_prabhasa_inference.ipynb):** MLM fill-in-the-blank inference and tokenization
- **[demo_02_mechanism_visualization.ipynb](demo_02_mechanism_visualization.ipynb):** Vidyut morpheme embeddings and Paribhāṣā masking visualizations
- **[demo_03_reproduce_blimp.ipynb](demo_03_reproduce_blimp.ipynb):** Minimal BLiMP evaluation demo

All notebooks are self-contained and run on free Colab (CPU or GPU).
```

### 1.3 Files to upload

From `/data/hf_export/arm_A_seed_0/`:
1. `config.json`
2. `model.safetensors`
3. `tokenizer.json`
4. `tokenizer_config.json`
5. `modeling_elc_psalm.py`
6. `configuration_elc_psalm.py`
7. `README.md` (model card, from template above)

### 1.4 Upload procedure

Using `huggingface_hub` CLI:

```bash
# Assuming HF_TOKEN is set with write access to qbz506 org

# Create repo
huggingface-cli repo create prabhasa-b_ss-0.1 --type model

# Clone and populate
git clone https://huggingface.co/qbz506/prabhasa-b_ss-0.1
cd prabhasa-b_ss-0.1
cp /path/to/hf_export/arm_A_seed_0/* .
git add .
git commit -m "Initial PRABHĀSA-B (Strict-Small) release"
git push

# Or, via huggingface_hub Python API:
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/data/hf_export/arm_A_seed_0",
    repo_id="qbz506/prabhasa-b_ss-0.1",
    repo_type="model",
)
```

### 1.5 Post-upload verification

```bash
# Test loading
python -c "
from transformers import AutoModelForMaskedLM, AutoTokenizer
m = AutoModelForMaskedLM.from_pretrained('qbz506/prabhasa-b_ss-0.1', trust_remote_code=True)
t = AutoTokenizer.from_pretrained('qbz506/prabhasa-b_ss-0.1')
print('✓ Model loaded successfully')
"

# Test in Colab
# Visit: https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/demo_01_prabhasa_inference.ipynb
# Update MODEL_ID = "qbz506/prabhasa-b_ss-0.1"
# Run and verify output
```

---

## Part 2: Dataset Publication (`qbz506/prabhasa-corpus-from-grammar`)

### 2.1 Dataset card template (datasets library)

```markdown
---
language:
  - en
license: cc-by-4.0
size_categories:
  - 10M<n<100M
task_categories:
  - language-modeling
task_ids:
  - language-modeling
---

# PRABHĀSA Corpus (from Grammar)

**Generated pretraining corpus** for PRABHĀSA experiments. BabyLM Strict-Small English subset (10M tokens, 4K types) augmented with Pāṇinian-inspired synthetic examples.

## Description

This is the **pretraining data** used to train PRABHĀSA-B (Strict-Small). It includes:

1. **BabyLM Strict-Small English corpus** (primary): ~10M tokens, 4K vocabulary
   - Sources: CHILDES (transcribed speech), Simple English Wikipedia, Children's Literature
   - Minimal preprocessing: lowercase, ASCII normalization, BPE tokenization (SentencePiece, 20K vocab)

2. **Optional Pāṇinian synthetic examples** (auxiliary, for H1_MECHANISM analysis):
   - Controlled minimal pairs for morphology, agreement, long-range dependencies
   - Generated from Pāṇinian grammar rules (dhātu roots, pratyaya affixes, sandhi)
   - Used for pre-pretraining (H1) and analysis; excluded from final 10M Strict-Small budget

## Structure

```
prabhasa-corpus-from-grammar/
  ├── babylm_ss_en.txt         # 10M tokens, line-delimited sentences
  ├── babylm_ss_en_ids.jsonl   # Tokenized (token ids), one doc per line
  ├── metadata.json
  ├── statistics.json            # Vocabulary counts, token statistics
  └── README.md
```

## Usage

### Load from Hugging Face Datasets

```python
from datasets import load_dataset

ds = load_dataset("qbz506/prabhasa-corpus-from-grammar", split="train")
print(ds[0])
# Output: {'text': 'the cat sleeps on the mat .'}

# Tokenize
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("qbz506/prabhasa-b_ss-0.1")

def tokenize_fn(batch):
    return tok(batch["text"], truncation=True, max_length=128)

ds_tok = ds.map(tokenize_fn, batched=True, batch_size=1000)
print(ds_tok[0])
# Output: {'input_ids': [101, 1045, 2572, ...], ...}
```

### Reproduce pretraining

```bash
cd PSALM
uv run psalm train \
  --config configs/pretraining/babylm_strict_small.yaml \
  --data data/babylm_ss_en_ids.jsonl \
  --output checkpoints/arm_B_mechanism_en_seed_0
```

## Statistics

- **Total tokens:** ~10M
- **Vocabulary size:** 20,000 (SentencePiece BPE)
- **Unique types:** 4,000 (approx, depends on tokenization)
- **Average sentence length:** ~15 tokens
- **Sources:**
  - CHILDES: ~40%
  - Simple English Wikipedia: ~30%
  - Children's Literature (GRETIL, Project Gutenberg): ~30%

## BabyLM reference

This dataset is derived from the **BabyLM challenge** (2023):
- Official benchmark: https://babylm.github.io/
- Paper: ["BabyLM: A cross-domain challenge for behavioral and developmental linguistics"](https://openreview.net/forum?id=dLrp48U6eN)
- Challenge track: Strict-Small (10M tokens, 4K vocabulary)

## License

- **BabyLM English:** CC-BY-4.0 (aggregated from open-source children's language corpora)
- **Pāṇinian synthetic examples:** CC-BY-4.0

## Citation

If you use this corpus, please cite:

1. This dataset:
   ```bibtex
   @dataset{sharaths2025prabhasa_corpus,
     title={PRABHĀSA Corpus (from Grammar)},
     author={Sharaths},
     year={2025},
     url={https://huggingface.co/datasets/qbz506/prabhasa-corpus-from-grammar}
   }
   ```

2. BabyLM benchmark:
   ```bibtex
   @inproceedings{warstadt2023babylm,
     title={BabyLM: A Cross-Domain Challenge for Behavioral and Developmental Linguistics},
     author={Warstadt, Alex and Mueller, Aaron and Choi, Yonatan},
     booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
     year={2023}
   }
   ```

## Warnings

- **Not suitable for:** World knowledge, fact-based question answering, knowledge-intensive tasks
- **Suitable for:** Pretraining, structural generalization experiments, compositional reasoning evaluation
- **Data quality:** BabyLM intentionally uses limited vocabulary and simple sentences; this is a design feature for studying sample efficiency, not a limitation of the source corpora.
```

### 2.2 Files to upload

```
data/
├── babylm_ss_en.txt
├── babylm_ss_en_ids.jsonl
├── metadata.json
└── statistics.json
```

Plus the dataset card (README.md).

### 2.3 Create repo and upload

```bash
huggingface-cli repo create prabhasa-corpus-from-grammar --type dataset

git clone https://huggingface.co/datasets/qbz506/prabhasa-corpus-from-grammar
cd prabhasa-corpus-from-grammar
cp /path/to/data/* .
git add .
git commit -m "Initial BabyLM Strict-Small corpus for PRABHĀSA pretraining"
git push
```

---

## Part 3: Colab Badge Integration

### 3.1 Generate Colab links

For each notebook, add the Colab badge at the top:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/demo_01_prabhasa_inference.ipynb)
```

The URL format is:
```
https://colab.research.google.com/github/{owner}/{repo}/blob/{branch}/notebooks/{notebook_name}.ipynb
```

Examples (update {owner}, {repo}, {branch}):
- **demo_01_prabhasa_inference.ipynb:**
  ```
  https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/demo_01_prabhasa_inference.ipynb
  ```
- **demo_02_mechanism_visualization.ipynb:**
  ```
  https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/demo_02_mechanism_visualization.ipynb
  ```
- **demo_03_reproduce_blimp.ipynb:**
  ```
  https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/demo_03_reproduce_blimp.ipynb
  ```

### 3.2 Add links to HF model card

In `qbz506/prabhasa-b_ss-0.1` README.md, include:

```markdown
## Demo Notebooks (Colab)

Click the badges to run on free Colab (CPU or GPU):

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/demo_01_prabhasa_inference.ipynb) **demo_01_prabhasa_inference.ipynb** — MLM fill-in-the-blank and tokenization
  
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/demo_02_mechanism_visualization.ipynb) **demo_02_mechanism_visualization.ipynb** — Morpheme embeddings and masking visualizations
  
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SharathSPhD/PSALM/blob/main/notebooks/demo_03_reproduce_blimp.ipynb) **demo_03_reproduce_blimp.ipynb** — BLiMP evaluation demo
```

---

## Part 4: Timeline & Checklist

### Pre-publication (before paper acceptance)

- [ ] Export final model to HF format
- [ ] Test model loading and inference (locally + Colab)
- [ ] Create and test demo notebooks
- [ ] Write model card and dataset card
- [ ] Review for citations, disclaimers, correctness
- [ ] Get team sign-off

### Publication day

- [ ] Create `qbz506/prabhasa-b_ss-0.1` model repo
- [ ] Create `qbz506/prabhasa-corpus-from-grammar` dataset repo
- [ ] Upload model files + README
- [ ] Upload dataset files + README
- [ ] Add Colab badges to model card and notebooks
- [ ] Verify model loads: `from transformers import ...` (test on free Colab)
- [ ] Update project website / GitHub README with links

### Post-publication

- [ ] Monitor HF Hub for downloads / feedback
- [ ] Create discussion threads on HF Hub for Q&A
- [ ] Link from paper / arXiv abstract
- [ ] Announce on social media (Twitter, etc.)

---

## Part 5: References

**Relevant docs:**
- Hugging Face Model Hub: https://huggingface.co/docs/hub/models-uploading
- Hugging Face Datasets: https://huggingface.co/docs/hub/datasets-uploading
- Model card guidelines: https://huggingface.co/docs/hub/model-cards
- Dataset card guidelines: https://huggingface.co/docs/hub/datasets-overview

**External references:**
- BabyLM challenge: https://babylm.github.io/
- SentencePiece: https://github.com/google/sentencepiece
- Transformers library: https://huggingface.co/transformers/

---

**Author:** Sharaths  
**Created:** 2025-06-05  
**Status:** Template (awaiting paper acceptance and final model export)
