#!/usr/bin/env bash
set -euo pipefail

NUM_QUESTIONS=10

for i in 0 1 2 3 4; do
  echo "=== student_model_$i ==="
  python eval/test_student_model.py \
    --student_model distill/out/grpo_distill/student_model_$i \
    --dataset data/wiki_20/data.json \
    --num_questions $NUM_QUESTIONS \
    --output eval/out_200/student_${i}_results.json
done

echo "=== base model (default --student_model) ==="
python eval/test_student_model.py \
  --dataset data/wiki_20/data.json \
  --num_questions $NUM_QUESTIONS \
  --output eval/out_200/student_base_results.json

python eval/plot_student_results.py --results-dir eval/out_200 --no-base

echo DONE_ALL
