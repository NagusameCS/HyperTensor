#!/bin/bash
# EC2 AUTO-PIPELINE: Math -> Language -> CECI
# Run on EC2: bash ~/pipeline.sh

set -e
PYTHON=/home/ubuntu/venv/bin/python3
LOG=~/pipeline.log

echo "========================================" | tee -a $LOG
echo "QWEN CECI PIPELINE START: $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG

# Step 1: Math LoRA
echo "[1/3] Training Math LoRA..." | tee -a $LOG
$PYTHON -u ~/train_qwen_pure.py --skill math --steps 4000 2>&1 | tee -a $LOG
echo "MATH DONE: $(date)" | tee -a $LOG

# Step 2: Language LoRA
echo "[2/3] Training Language LoRA..." | tee -a $LOG
$PYTHON -u ~/train_qwen_pure.py --skill language --steps 4000 2>&1 | tee -a $LOG
echo "LANGUAGE DONE: $(date)" | tee -a $LOG

# Step 3: CECI Splice
echo "[3/3] CECI Splice at k=768..." | tee -a $LOG
$PYTHON -u ~/ceci_qwen.py 2>&1 | tee -a $LOG
echo "CECI DONE: $(date)" | tee -a $LOG

echo "========================================" | tee -a $LOG
echo "PIPELINE COMPLETE: $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG
