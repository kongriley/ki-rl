#!/bin/bash
# Run ablation experiments. Each experiment gets its own output dir and result name.
#
# Usage:
#   ./run_ablations.sh                     # run all ablations locally (sequential)
#   ./run_ablations.sh --slurm             # submit each as a separate SLURM job
#   ./run_ablations.sh --list              # just print what would run
#   DATASET=thetech ./run_ablations.sh     # override dataset

set -euo pipefail

DATASET="${DATASET:-wiki_20}"
NUM_ITERATIONS="${NUM_ITERATIONS:-30}"
SLURM=false
LIST_ONLY=false
LOG_DIR="./out/logs/"

for arg in "$@"; do
    case "$arg" in
        --slurm) SLURM=true ;;
        --list)  LIST_ONLY=true ;;
    esac
done

source .venv/bin/activate
mkdir -p "$LOG_DIR"

# ── Common args shared by all experiments ──
common_args=(
    --dataset_path "./data/${DATASET}/data.json"
    --num_generation_iterations "${NUM_ITERATIONS}"
    --num_question_generations 5
    --num_questions_per_generation 5
    --report_student_performance
    --eval_questions_path "./data/${DATASET}/gpt-5-mini_questions.jsonl"
    --eval_judge_backend openai
    --eval_judge_model gpt-5-mini
    --save_student_result_copy_dir "./results/${DATASET}"
    --num_distill_epochs 1
    --grpo_beta 0.001
)

# ── Define experiments ──
# Format: name|extra_args (pipe-separated)
experiments=(
    # Our method (default config with accumulation + good questions)
    # "student_model|--use_good_questions --accumulate_questions"

    # Ablation: no good question pooling
    "no_good_q|--no-use_good_questions --accumulate_questions"

    # Ablation: no question accumulation
    "no_accum|--use_good_questions"

    # Ablation: no good questions AND no accumulation
    "no_good_q_no_accum|--no-use_good_questions"

    # Ablation: skip generator training (use base model for question generation)
    # "no_gen|--skip_generator_training --use_good_questions --accumulate_questions"

    # Diversity penalty sweep
    "diversity_0.05|--use_good_questions --accumulate_questions --diversity_penalty_weight 0.05"
    "diversity_0.1|--use_good_questions --accumulate_questions --diversity_penalty_weight 0.1"
    "diversity_0.5|--use_good_questions --accumulate_questions --diversity_penalty_weight 0.5"

    # Different model (uncomment and adjust as needed)
    # "llama3_8b|--use_good_questions --accumulate_questions --model_name meta-llama/Llama-3.1-8B-Instruct"
)

run_experiment() {
    local name="$1"
    local extra_args="$2"

    local output_dir="./distill/out/ablation_${name}"
    local result_name="ablation_${name}"

    echo "=== Experiment: ${name} ==="
    echo "  Output: ${output_dir}"
    echo "  Extra args: ${extra_args}"

    if $LIST_ONLY; then
        return
    fi

    # shellcheck disable=SC2086
    if $SLURM; then
        local log="/data/scratch/rileyis/runs/ki-rl/ablation_${name}.%j.out"
        sbatch --job-name="abl_${name}" \
               --partition=vision-pulkitag-h100 \
               --qos=vision-pulkitag-debug \
               --account=vision-pulkitag-urops \
               --gres=gpu:1 \
               --cpus-per-task=16 \
               --mem=100G \
               --time=6:00:00 \
               --export=ALL \
               --output="$log" \
               --wrap="cd /data/scratch/rileyis/ki-rl && source .venv/bin/activate && python -u distill/main.py ${common_args[*]} --output_dir ${output_dir} --save_student_result_copy_name ${result_name} ${extra_args}"
    else
        local log="${LOG_DIR}/ablation_${name}_$(date +%Y%m%d_%H%M%S).out"
        python -u distill/main.py \
            "${common_args[@]}" \
            --output_dir "${output_dir}" \
            --save_student_result_copy_name "${result_name}" \
            ${extra_args} \
            > >(tee "$log") 2>&1
    fi
}

echo "Dataset: ${DATASET}"
echo "Iterations: ${NUM_ITERATIONS}"
echo "Mode: $( $SLURM && echo 'SLURM' || echo 'local' )"
echo ""

for exp in "${experiments[@]}"; do
    IFS='|' read -r name extra_args <<< "$exp"
    run_experiment "$name" "$extra_args"
done

echo ""
echo "Done. Results will be saved to ./results/${DATASET}/"
