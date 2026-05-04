#!/bin/bash
# ISAGI DUEL MARATHON — Run all 6 tasks for 1000 turns each on EC2
# Usage: bash duel_marathon.sh

MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTDIR="/home/ubuntu/benchmarks/duel_marathon"
mkdir -p "$OUTDIR"

TASKS=("math_proof" "code_review" "creative_writing" "system_design" "puzzle_chain" "research_proposal")
TURNS=1000

echo "============================================"
echo "  ISAGI DUEL MARATHON"
echo "  $(date)"
echo "  Model: $MODEL"
echo "  Tasks: ${#TASKS[@]} | Turns per task: $TURNS"
echo "  Total turns: $((${#TASKS[@]} * TURNS))"
echo "============================================"

for TASK in "${TASKS[@]}"; do
    echo ""
    echo "=== Starting task: $TASK ==="
    echo "  $(date)"
    
    SAVE_PATH="$OUTDIR/duel_${TASK}_${TURNS}t.json"
    
    python3 scripts/isagi_duel.py \
        --task "$TASK" \
        --turns "$TURNS" \
        --model "$MODEL" \
        --4bit \
        --save "$SAVE_PATH" \
        2>&1 | tee "$OUTDIR/duel_${TASK}.log"
    
    echo "  Completed: $TASK"
    echo "  $(date)"
    
    # Sleep to let GPU cool down
    sleep 30
done

echo ""
echo "============================================"
echo "  MARATHON COMPLETE"
echo "  $(date)"
echo "  Output: $OUTDIR"
echo "============================================"

# Summary
echo ""
echo "=== SUMMARY ==="
for TASK in "${TASKS[@]}"; do
    RESULT_FILE="$OUTDIR/duel_${TASK}_${TURNS}t.json"
    if [ -f "$RESULT_FILE" ]; then
        ANALYSIS=$(python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
a = data['analysis']
print(f\"  {a['task']:25s} | turns={a['n_turns']:4d} | expand={a['isagi_expansion_rate']:5.1f}% | gtc={a['isagi_gtc_hit_rate']:5.1f}% | metric={a['metric_final']:.4f} | success={a['success_estimate']}/100\")
" 2>/dev/null)
        echo "$ANALYSIS"
    else
        echo "  $TASK: NO RESULTS YET"
    fi
done
