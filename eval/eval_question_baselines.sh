MODE="base" # base, icl, rag

python eval/eval_questions.py \
    $(if [[ "$MODE" == "icl" || "$MODE" == "rag" ]]; then echo --${MODE}; fi) \
    --questions_path data/wiki_20/gpt-5-mini_questions.jsonl \
    --data_path data/wiki_20/data.json \
    --output results/results_${MODE}.json \
    --judge_model gpt-5-mini \
    --judge_backend openai \
    --batch_judge