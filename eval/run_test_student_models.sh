#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?Usage: $0 <questions_file | question_model> [question_backend]}"
QUESTION_BACKEND="${2:-hf}"
NUM_QUESTIONS=10
JUDGE_MODEL="gpt-5-mini"
JUDGE_BACKEND="openai"
OUTPUT_DIR="eval/out_gpt-5-mini"

if [[ -f "$MODE" ]]; then
  QUESTION_ARGS=(--questions_path "$MODE")
else
  QUESTION_ARGS=(--question_model "$MODE" --question_backend "$QUESTION_BACKEND" --num_questions "$NUM_QUESTIONS")
fi

JUDGE_ARGS=()
if [[ -n "$JUDGE_MODEL" ]]; then
  JUDGE_ARGS+=(--judge_model "$JUDGE_MODEL")
fi
if [[ -n "$JUDGE_BACKEND" ]]; then
  JUDGE_ARGS+=(--judge_backend "$JUDGE_BACKEND")
fi

for i in 0 1 2 3 4; do
  echo "=== student_model_$i ==="
  python eval/eval_questions.py \
    --model distill/out/grpo_distill/student_model_$i \
    --data_path data/wiki_20/data.json \
    "${QUESTION_ARGS[@]}" \
    "${JUDGE_ARGS[@]}" \
    --output $OUTPUT_DIR/student_${i}_results.json \
    --batch_judge
done

echo "=== base model ==="
python eval/eval_questions.py \
  --data_path data/wiki_20/data.json \
  "${QUESTION_ARGS[@]}" \
  "${JUDGE_ARGS[@]}" \
  --output $OUTPUT_DIR/student_base_results.json \
  --batch_judge

python eval/plot_student_results.py --results-dir $OUTPUT_DIR --no-base

echo DONE_ALL
