#!/bin/bash
# ONE-SHOT SCRIPT: Run this on Runpod and walk away.
# It will: clone repo, install deps, download data, train tokenizer,
# generate data shards, smoke test, and run full 8xH100 training.
#
# Usage: curl -sSL <raw-github-url> | bash
# Or: paste the contents into the Runpod terminal
set -euo pipefail

echo "============================================="
echo " Parameter Golf - Full Pipeline"
echo " Started: $(date)"
echo "============================================="

cd /workspace

# 1. Clone repo
if [ ! -d "parameter-golf" ]; then
    git clone https://github.com/openai/parameter-golf.git
fi
cd parameter-golf

# 2. Install deps
pip install -q sentencepiece huggingface_hub

# 3. Download SP-4096 data if available, otherwise generate it
echo ""
echo "=== Downloading data ==="

# First try downloading SP-4096 directly from HF
python data/cached_challenge_fineweb.py --variant sp4096 --train-shards 80 2>&1 && {
    echo "SP-4096 data downloaded from HuggingFace"
    SP4096_READY=1
} || {
    echo "SP-4096 not on HF, will generate from docs..."
    SP4096_READY=0
}

if [ "$SP4096_READY" -eq 0 ]; then
    echo "=== Downloading raw docs + SP-1024 for tokenizer training ==="
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 0 --with-docs

    echo "=== Training SP-4096 tokenizer and generating shards ==="
    python data/download_hf_docs_and_tokenize.py \
        --output-root data \
        --tokenizer-config data/tokenizer_specs_sp4096.json \
        --skip-byte
fi

# Verify data exists
echo ""
echo "=== Data verification ==="
ls -lh data/tokenizers/fineweb_4096_bpe.model 2>/dev/null || ls -lh data/tokenizers/*4096* 2>/dev/null || echo "WARNING: SP-4096 tokenizer not found"
TRAIN_SHARDS=$(ls data/datasets/fineweb10B_sp4096/fineweb_train_*.bin 2>/dev/null | wc -l)
VAL_SHARDS=$(ls data/datasets/fineweb10B_sp4096/fineweb_val_*.bin 2>/dev/null | wc -l)
echo "Train shards: $TRAIN_SHARDS, Val shards: $VAL_SHARDS"

if [ "$TRAIN_SHARDS" -eq 0 ] || [ "$VAL_SHARDS" -eq 0 ]; then
    echo "ERROR: Data shards missing. Falling back to SP-1024..."
    # Fallback: download SP-1024 and use that
    python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
    # Override env vars to use SP-1024
    export DATA_PATH="./data/datasets/fineweb10B_sp1024"
    export TOKENIZER_PATH="./data/tokenizers/fineweb_1024_bpe.model"
    export VOCAB_SIZE=1024
fi

# 4. Copy our improved training script to working directory
cp records/track_10min_16mb/2026-03-19_Improved/train_gpt.py ./train_improved.py

echo ""
echo "=== Starting training ==="
echo "GPUs available: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 5. Run training (8 GPU)
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Running with $NUM_GPUS GPUs..."
echo "Training started: $(date)"

torchrun --nproc_per_node=$NUM_GPUS ./train_improved.py 2>&1 | tee training_log.txt

echo ""
echo "============================================="
echo " Training complete: $(date)"
echo "============================================="

# 6. Show results
echo ""
echo "=== RESULTS ==="
grep -E "val_bpb|val_loss|submission size|Serialized|stopping" training_log.txt | tail -20
echo ""
ls -lh final_model.int8.ptz 2>/dev/null
ls -lh final_model.pt 2>/dev/null

echo ""
echo "=== Key files to download ==="
echo "  /workspace/parameter-golf/final_model.int8.ptz"
echo "  /workspace/parameter-golf/training_log.txt"
echo "  /workspace/parameter-golf/train_improved.py"
echo ""
echo "You can now terminate the pod."
