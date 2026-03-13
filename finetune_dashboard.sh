#!/usr/bin/env bash
# Monitor GLM-OCR fine-tuning progress on Longleaf.
# Usage: watch -n 30 ./finetune_dashboard.sh

USER_NAME="ncaren"
WORK="/work/users/n/c/ncaren"
LORA_DIR="$WORK/glm-finetune/output/lora"
MERGED_DIR="$WORK/glm-finetune/output/merged"

# Safer terminal setting for clusters
export TERM="${TERM/xterm-ghostty/xterm-256color}"

clear
echo "===== GLM-OCR FINE-TUNE DASHBOARD ====="
date
echo

# --- Job status ---
running=$(squeue -u "$USER_NAME" -h -t RUNNING -n glm_finetune,glm_eval 2>/dev/null | wc -l | tr -d ' ')
pending=$(squeue -u "$USER_NAME" -h -t PENDING -n glm_finetune,glm_eval 2>/dev/null | wc -l | tr -d ' ')
printf "Running jobs:    %s\n" "$running"
printf "Pending jobs:    %s\n" "$pending"

echo
echo "----- ACTIVE JOBS -----"
squeue -u "$USER_NAME" -n glm_finetune,glm_eval 2>/dev/null || echo "(none)"

# --- Training progress (parse log file) ---
echo
echo "----- TRAINING PROGRESS -----"
LOGFILE=$(ls -t finetune_*.out 2>/dev/null | head -1)
if [[ -n "$LOGFILE" ]]; then
    # Show GPU info
    grep -m1 "NVIDIA\|Tesla\|A100\|L40" "$LOGFILE" 2>/dev/null

    # Extract latest training loss line (LLaMA-Factory logs loss at each logging_step)
    last_loss=$(grep -oP "'loss':\s*[\d.]+" "$LOGFILE" 2>/dev/null | tail -1)
    last_step=$(grep -oP "'current_steps':\s*\d+" "$LOGFILE" 2>/dev/null | tail -1)
    total_steps=$(grep -oP "'total_steps':\s*\d+" "$LOGFILE" 2>/dev/null | tail -1)
    epoch=$(grep -oP "'epoch':\s*[\d.]+" "$LOGFILE" 2>/dev/null | tail -1)

    # Also try HF Trainer format
    if [[ -z "$last_loss" ]]; then
        last_loss=$(grep -oP "'loss': [\d.]+" "$LOGFILE" 2>/dev/null | tail -1)
    fi
    if [[ -z "$last_loss" ]]; then
        last_loss=$(grep -oP "loss.*?[\d]+\.[\d]+" "$LOGFILE" 2>/dev/null | tail -1)
    fi

    if [[ -n "$last_step" && -n "$total_steps" ]]; then
        step_num=$(echo "$last_step" | grep -oP '\d+')
        total_num=$(echo "$total_steps" | grep -oP '\d+')
        if [[ "$total_num" -gt 0 ]]; then
            pct=$(awk "BEGIN {printf \"%.1f\", 100 * $step_num / $total_num}")
            printf "Step:            %s / %s (%s%%)\n" "$step_num" "$total_num" "$pct"
        fi
    fi
    if [[ -n "$epoch" ]]; then
        printf "Epoch:           %s\n" "$(echo "$epoch" | grep -oP '[\d.]+')"
    fi
    if [[ -n "$last_loss" ]]; then
        printf "Loss:            %s\n" "$(echo "$last_loss" | grep -oP '[\d.]+')"
    fi

    # Show loss trend (last 5 logged values)
    echo
    echo "  Loss trend (last 5):"
    grep -oP "'loss':\s*[\d.]+" "$LOGFILE" 2>/dev/null | tail -5 | while read -r line; do
        val=$(echo "$line" | grep -oP '[\d.]+')
        printf "    %s\n" "$val"
    done

    # Check for errors
    ERRFILE="${LOGFILE/.out/.err}"
    if [[ -f "$ERRFILE" ]]; then
        err_count=$(grep -ciP "error|exception|traceback|oom|killed" "$ERRFILE" 2>/dev/null || echo 0)
        if [[ "$err_count" -gt 0 ]]; then
            echo
            echo "  ⚠ $err_count error(s) in $ERRFILE:"
            grep -iP "error|exception|oom|killed" "$ERRFILE" 2>/dev/null | tail -3
        fi
    fi
else
    echo "(no finetune log files found)"
fi

# --- Checkpoints ---
echo
echo "----- CHECKPOINTS -----"
if [[ -d "$LORA_DIR" ]]; then
    checkpoints=$(ls -d "$LORA_DIR"/checkpoint-* 2>/dev/null | wc -l | tr -d ' ')
    printf "LoRA checkpoints: %s\n" "$checkpoints"
    ls -dt "$LORA_DIR"/checkpoint-* 2>/dev/null | head -3 | while read -r d; do
        printf "  %s  (%s)\n" "$(basename "$d")" "$(date -r "$d" '+%H:%M:%S' 2>/dev/null || stat -c '%y' "$d" 2>/dev/null | cut -d. -f1)"
    done
else
    echo "(no checkpoints yet)"
fi

# --- Merged model ---
if [[ -d "$MERGED_DIR" ]]; then
    echo
    echo "  ✓ Merged model exists at $MERGED_DIR"
fi

# --- Evaluation progress ---
EVALLOG=$(ls -t eval_*.out 2>/dev/null | head -1)
if [[ -n "$EVALLOG" ]]; then
    echo
    echo "----- EVALUATION PROGRESS -----"
    # Show which model is being evaluated
    current_model=$(grep -oP "Running \K.*" "$EVALLOG" 2>/dev/null | tail -1)
    if [[ -n "$current_model" ]]; then
        printf "Current:         %s\n" "$current_model"
    fi

    # Count completed transcriptions
    base_done=$(ls "$WORK/Inkbench/ocr-results/glm-ocr-base/"*.txt 2>/dev/null | wc -l | tr -d ' ')
    ft_done=$(ls "$WORK/Inkbench/ocr-results/glm-ocr-finetuned/"*.txt 2>/dev/null | wc -l | tr -d ' ')
    printf "Base model:      %s / 400 images\n" "$base_done"
    printf "Fine-tuned:      %s / 400 images\n" "$ft_done"

    # Show accuracy results if available
    ACCURACY_CSV="$WORK/Inkbench/ocr_eval_model_accuracy.csv"
    if [[ -f "$ACCURACY_CSV" ]]; then
        echo
        echo "----- RESULTS -----"
        column -t -s, "$ACCURACY_CSV" 2>/dev/null || cat "$ACCURACY_CSV"
    fi
fi

# --- Last 5 lines of latest log ---
echo
echo "----- LATEST LOG -----"
LATEST=$(ls -t finetune_*.out eval_*.out 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    echo "($LATEST)"
    tail -5 "$LATEST"
else
    echo "(no log files)"
fi
