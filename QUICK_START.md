# Quick Start: prabhasa-b_s Corpus Training

## One-Liner Status

**TASK A:** 9.74M words tokenized (extend to 100M with one command)  
**TASK B:** 945 grammar-derived Sanskrit forms ready to use  
**STATUS:** Ready to train immediately

## Essential Paths

```
English corpus:    /home/sharaths/projects/PSALM-integration/data/corpora/strict/english_base.bin
Dose corpus:       /home/sharaths/projects/PSALM-integration/data/corpora/strict/dose_grammar.bin
Tokenizer:         /home/sharaths/projects/PSALM-integration/data/tokenizer/strict_small/spm.model
```

## Download Full 100M English (Optional)

```bash
cd /home/sharaths/projects/PSALM-integration
uv run python scripts/prepare_babylm_100m.py --force
# Takes ~30-60 min, produces 200 MB binary file
```

## Launch Training (Seed 0)

```bash
cd /home/sharaths/projects/PSALM-integration
uv run python src/psalm/application/train.py \
  --config configs/prabhasa_b_s_small.yaml \
  --english-bin data/corpora/strict/english_base.bin \
  --dose-bin data/corpora/strict/dose_grammar.bin \
  --output-dir runs/prabhasa_b_s_seed_0 \
  --seed 0
```

## Launch 5-Seed Run (Statistical Validity)

```bash
cd /home/sharaths/projects/PSALM-integration
for seed in 0 1 2 3 4; do
  uv run python src/psalm/application/train.py \
    --config configs/prabhasa_b_s_small.yaml \
    --english-bin data/corpora/strict/english_base.bin \
    --dose-bin data/corpora/strict/dose_grammar.bin \
    --output-dir runs/prabhasa_b_s_seed_${seed} \
    --seed $seed
done
```

## Verify Data

```bash
cd /home/sharaths/projects/PSALM-integration

# Check English corpus
wc -w data/corpora/strict/english_base.txt

# Check dose corpus
wc -l data/corpora/strict/dose_grammar.txt

# Verify tokenization
uv run python -c "import numpy as np; data = np.memmap('data/corpora/strict/english_base.bin', dtype='uint16'); print(f'English tokens: {len(data):,}')"
uv run python -c "import numpy as np; data = np.memmap('data/corpora/strict/dose_grammar.bin', dtype='uint16'); print(f'Dose tokens: {len(data):,}')"

# Regenerate manifest
uv run python scripts/corpus_manifest_gen.py --output corpus_manifest.json
```

## Documentation

- **CORPUS_PREP_SUMMARY.md** — Full technical guide
- **TASK_COMPLETION_REPORT.md** — Executive summary
- **corpus_manifest.json** — Machine-readable state

## Expected Runtime

- Per seed: 10-11 hours (DGX Spark GB10)
- 5 seeds: 50-55 hours (parallelizable)

## Key Facts

- English: 9.74M words (can scale to 100M)
- Dose: 945 verb forms, 210.7K tokens
- Format: uint16 memmap (zero-copy, 10x throughput)
- Tokenizer: SentencePiece (vocab=20000, SLP1)
- Budget: Dose NOT counted toward 100M (per ADR-0020)
- Status: Ready for immediate training

## Commit

```
a8fac73: feat(data): BabyLM 100M corpus prep for prabhasa-b_s track
```
