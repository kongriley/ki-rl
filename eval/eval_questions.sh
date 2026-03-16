python eval/eval_questions.py \
  --backend hf \
  --model allenai/OLMo-2-1124-7B-Instruct \
  --judge_backend openai \
  --judge_model gpt-5-mini \
  --data_path data/wiki_20/data.json \
  --questions_path data/wiki_20/gpt-5-mini_questions.jsonl \
  --icl \
  --output eval/out/results_icl.json
