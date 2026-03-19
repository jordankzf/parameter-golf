#!/bin/bash
# Run this on Runpod after cloning the repo
# Usage: bash setup_runpod.sh
set -e

echo "=== Parameter Golf - Runpod Setup ==="

# 1. Install dependencies
pip install sentencepiece huggingface_hub

# 2. Download data - SP-1024 (for training, since we need to generate SP-4096)
cd /workspace/parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80 --with-docs

# 3. Train SP-4096 tokenizer on the raw docs
echo "=== Training SP-4096 tokenizer ==="
python data/download_hf_docs_and_tokenize.py \
    --output-root data \
    --tokenizer-config data/tokenizer_specs_sp4096.json \
    --skip-byte

echo "=== Verifying tokenizer and data ==="
ls -lh data/tokenizers/fineweb_4096_bpe.model
ls -lh data/datasets/fineweb10B_sp4096/

# 4. Quick smoke test (CPU, 2 steps)
echo "=== CPU smoke test ==="
CUDA_VISIBLE_DEVICES="" \
MAX_WALLCLOCK_SECONDS=30 \
ITERATIONS=2 \
WARMUP_STEPS=0 \
VAL_LOSS_EVERY=1 \
python records/track_10min_16mb/2026-03-19_Improved/train_gpt.py 2>&1 || echo "CPU test skipped (needs CUDA)"

echo "=== Setup complete ==="
echo ""
echo "To run training:"
echo "  torchrun --nproc_per_node=8 records/track_10min_16mb/2026-03-19_Improved/train_gpt.py"
