#!/bin/bash
# ==============================================
# Transformer Seq2Seq 训练脚本
# 环境要求：
#   - Python ≥ 3.8
#   - PyTorch ≥ 2.0
#   - CUDA ≥ 11.8
#   - GPU: NVIDIA RTX 3090 (24GB 显存推荐)
# ==============================================

# ----------- 基本设置 -----------
EPOCHS=20
LR=5e-4
BATCH_SIZE=64
SAVE_DIR="checkpoints"
WARMUP_STEPS=4000
MAX_STEPS=50000
MAX_SEQ_LEN=100
SEED=42

# ----------- 路径设置 -----------
PROJECT_ROOT="$(dirname $(dirname "$0"))"
TRAIN_SCRIPT="$PROJECT_ROOT/src/train.py"
DATA_DIR="$PROJECT_ROOT/data/processed"
TOKENIZER="$PROJECT_ROOT/data/tokenizer/iwslt_bpe.model"

# ----------- 环境检测 -----------
echo "🔥 Checking environment..."
python -c "import torch; print('✅ PyTorch version:', torch.__version__)"
python -c "import sentencepiece; print('✅ SentencePiece version:', sentencepiece.__version__)"

# ----------- 启动训练 -----------
echo "🚀 Starting training..."
python "$TRAIN_SCRIPT" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --save-dir $SAVE_DIR \
    --warmup-steps $WARMUP_STEPS \
    --max-steps $MAX_STEPS \
    --max-seq-len $MAX_SEQ_LEN \
    --seed $SEED

echo "✅ Training finished!"
