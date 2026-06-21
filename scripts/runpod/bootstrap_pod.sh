#!/usr/bin/env bash
# Full PULL-ONLY pod provisioner (no agent egress). Piped via run_pod.sh bootstrap.
# Writes a detached provisioner to /workspace/_provision.sh, nohups it, returns immediately.
# Monitor /workspace/provision.log for "PROVISION_DONE".
set -uo pipefail
cat > /workspace/_provision.sh <<'PROV'
#!/usr/bin/env bash
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER=0
cd /workspace
BR=feature/v0.2-arch-sweep

echo "=== [1/6] code via git (clean public branch) ==="
if [ -d psalm/.git ]; then (cd psalm && git fetch --depth 1 origin "$BR" && git reset --hard origin/"$BR"); \
else rm -rf psalm && git clone --branch "$BR" --depth 1 https://github.com/SharathSPhD/prabhasa-babylm.git psalm; fi
[ -d panini-data-toolkit/.git ] || git clone --depth 1 https://github.com/SharathSPhD/panini-data-toolkit.git panini-data-toolkit
cd /workspace/psalm

echo "=== [2/6] env (uv sync --extra ml --extra stats) ==="
uv sync --extra ml --extra stats 2>&1 | tail -3
echo "=== [3/6] spaCy model ==="
uv run --no-sync python -m spacy download en_core_web_sm 2>&1 | tail -1 || echo "[warn] spacy dl"
uv run --no-sync python -c "import torch,psalm,spacy; print('env OK torch',torch.__version__,'cuda',torch.cuda.is_available())"

echo "=== [4/6] eval pipeline (clone + 192 fix) ==="
[ -d vendor/babylm-evaluation-pipeline-2026 ] || git clone --depth 1 https://github.com/babylm-org/babylm-eval.git vendor/babylm-evaluation-pipeline-2026
sed -i 's/--sequence_length 512/--sequence_length 192/g' vendor/babylm-evaluation-pipeline-2026/strict/scripts/eval_finetuning.sh 2>/dev/null || true

echo "=== [5/6] reproduce STRICT corpus (pull public BabyLM + tokenize) ==="
mkdir -p data/corpora/strict
uv run --no-sync python scripts/prepare_babylm_100m.py 2>&1 | tail -8

echo "=== [5b/6] pull EXPANSION from public HF (Sanskrit/Paribhāṣā dose + grammar; v0.1 release) ==="
mkdir -p docs/data data/corpora/strict_small/arms data/corpora/grammar
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('qbz506/psalm-corpora', repo_type='dataset', local_dir='/tmp/psc')" 2>&1 | tail -1
cp -f /tmp/psc/strict-small-arms.json docs/data/strict-small-arms.json
cp -f /tmp/psc/dose_*.txt data/corpora/strict_small/arms/
uv run --no-sync python -c "from huggingface_hub import snapshot_download; snapshot_download('qbz506/prabhasa-babylm-grammar', repo_type='dataset', local_dir='data/corpora/grammar')" 2>&1 | tail -1
echo "expansion: $(ls docs/data/strict-small-arms.json data/corpora/strict_small/arms/*.txt 2>/dev/null | wc -l) files placed"

echo "=== [6/6] verify corpus byte-size matches GB10 (strict english_base.bin == 313669342) ==="
SZ=$(stat -c %s data/corpora/strict/english_base.bin 2>/dev/null || echo 0)
echo "strict english_base.bin size = $SZ (expect 313669342)"
if [ "$SZ" = "313669342" ]; then echo "CORPUS_MATCH_OK"; else echo "CORPUS_MATCH_MISMATCH (size $SZ)"; fi
echo "PROVISION_DONE"
PROV
chmod +x /workspace/_provision.sh
nohup bash /workspace/_provision.sh > /workspace/provision.log 2>&1 < /dev/null &
echo "PROVISION_LAUNCHED pid=$!"
