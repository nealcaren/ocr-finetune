#!/usr/bin/env bash
# Monitor OCR benchmarking and fine-tuning progress on Longleaf.
# Usage: watch -n 30 ./finetune_dashboard.sh

USER_NAME="ncaren"
WORK="/work/users/n/c/ncaren"
RESULTS_DIR="$WORK/Inkbench/ocr-results"
EVAL_DIR="$WORK/Inkbench/ocr-eval"
LORA_DIR="$WORK/glm-finetune/output/lora"
MERGED_DIR="$WORK/glm-finetune/output/merged"

# All benchmark models
MODELS=(olmocr nanonets-ocr2 chandra dots-ocr deepseek-ocr2 rolmocr minicpm-v-4.5 glm-ocr-base glm-ocr-finetuned qwen3-vl-8b)

# Safer terminal setting for clusters
export TERM="${TERM/xterm-ghostty/xterm-256color}"

clear
echo "===== OCR BENCHMARK DASHBOARD ====="
date
echo

# --- Job status (only our jobs, exclude unrelated ones like ocr_all) ---
echo "----- ACTIVE JOBS -----"
squeue -u "$USER_NAME" -o "%.10i %.20j %.8T %.10M %.6D %R" 2>/dev/null \
    | grep -E "JOBID|ocr_bench|glm_finetune|glm_eval" | head -20
running=$(squeue -u "$USER_NAME" -h -t RUNNING 2>/dev/null | grep -E "ocr_bench|glm_finetune|glm_eval" | wc -l | tr -d ' ')
pending=$(squeue -u "$USER_NAME" -h -t PENDING 2>/dev/null | grep -E "ocr_bench|glm_finetune|glm_eval" | wc -l | tr -d ' ')
echo
printf "Running: %s   Pending: %s\n" "$running" "$pending"

# --- Benchmark progress per model ---
echo
echo "----- BENCHMARK PROGRESS -----"
printf "%-20s %8s %8s %s\n" "Model" "Images" "/ 400" "Eval"
printf "%-20s %8s %8s %s\n" "-----" "------" "-----" "----"
for m in "${MODELS[@]}"; do
    model_dir="$RESULTS_DIR/$m"
    eval_csv="$EVAL_DIR/$m.csv"
    if [[ -d "$model_dir" ]]; then
        count=$(ls "$model_dir"/*.txt 2>/dev/null | wc -l | tr -d ' ')
        if [[ -f "$eval_csv" ]]; then
            acc=$(awk -F, 'NR>1 && $4=="ok" && $7!="" {sum+=$7; n++} END {if(n>0) printf "%.1f%%", (1-sum/n)*100}' "$eval_csv" 2>/dev/null)
            eval_status="${acc:-done}"
        else
            eval_status="-"
        fi
        printf "%-20s %8s %8s %s\n" "$m" "$count" "/ 400" "$eval_status"
    else
        printf "%-20s %8s %8s %s\n" "$m" "-" "/ 400" "-"
    fi
done

# --- Per-model eval results if available ---
eval_csvs=$(ls "$EVAL_DIR"/*.csv 2>/dev/null)
if [[ -n "$eval_csvs" ]]; then
    echo
    echo "----- COMPLETED RESULTS -----"
    printf "%-20s %10s %10s %10s %6s\n" "Model" "Accuracy" "CER_alnum" "WER" "n"
    printf "%-20s %10s %10s %10s %6s\n" "-----" "--------" "---------" "---" "-"
    for csv_file in $eval_csvs; do
        m=$(basename "$csv_file" .csv)
        result=$(awk -F, 'NR>1 && $4=="ok" && $7!="" {cer+=$7; wer+=$5; n++} END {
            if(n>0) printf "%.3f,%.4f,%.4f,%d", 1-cer/n, cer/n, wer/n, n
        }' "$csv_file" 2>/dev/null)
        if [[ -n "$result" ]]; then
            IFS=',' read -r acc cer wer n <<< "$result"
            printf "%-20s %10s %10s %10s %6s\n" "$m" "$acc" "$cer" "$wer" "$n"
        fi
    done
fi

# --- Fine-tuning checkpoints (if training) ---
if [[ -d "$LORA_DIR" ]]; then
    checkpoints=$(ls -d "$LORA_DIR"/checkpoint-* 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$checkpoints" -gt 0 ]]; then
        echo
        echo "----- FINE-TUNE CHECKPOINTS -----"
        printf "LoRA checkpoints: %s\n" "$checkpoints"
        ls -dt "$LORA_DIR"/checkpoint-* 2>/dev/null | head -3 | while read -r d; do
            printf "  %s  (%s)\n" "$(basename "$d")" "$(date -r "$d" '+%H:%M:%S' 2>/dev/null || stat -c '%y' "$d" 2>/dev/null | cut -d. -f1)"
        done
    fi
fi
if [[ -d "$MERGED_DIR" ]]; then
    echo "  ✓ Merged model at $MERGED_DIR"
fi

# --- Errors in recent logs ---
echo
echo "----- RECENT ERRORS -----"
err_found=0
for errfile in $(ls -t benchmark_*.err finetune_*.err eval_*.err 2>/dev/null | head -8); do
    err_count=$(grep -iP "error|exception|traceback|oom|killed" "$errfile" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$err_count" -gt 0 ]]; then
        echo "  ⚠ $errfile ($err_count errors):"
        grep -iP "error|exception|oom|killed" "$errfile" 2>/dev/null | tail -2
        echo
        err_found=1
    fi
done
if [[ "$err_found" -eq 0 ]]; then
    echo "  (none)"
fi

# --- Last lines of latest log ---
echo
echo "----- LATEST LOG -----"
LATEST=$(ls -t benchmark_*.out finetune_*.out eval_*.out 2>/dev/null | head -1)
if [[ -n "$LATEST" ]]; then
    echo "($LATEST)"
    tail -5 "$LATEST"
else
    echo "(no log files)"
fi
