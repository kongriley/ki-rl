MODE="${1:-base}" # usage: ./eval/eval_question_baselines.sh [base|icl|rag|cpt]
DATASET="thetech"

python eval/eval_questions.py \
    $(if [[ "$MODE" == "icl" || "$MODE" == "rag" ]]; then echo --${MODE}; fi) \
    $(if [[ "$MODE" == "cpt" ]]; then echo --model out/cpt_${DATASET}; fi) \
    --questions_path data/${DATASET}/gpt-5-mini_questions.jsonl \
    --data_path data/${DATASET}/data.json \
    --output results/${DATASET}/results_${MODE}.json \
    --judge_model gpt-5-mini \
    --judge_backend openai \
    --batch_judge
