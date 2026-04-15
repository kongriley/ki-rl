LOG_DIR="./out/"
DEBUG=

python -u distill/main.py \
	--dataset_path ./data/wiki_20/single_passage.json \
	--output_dir ./distill/out/grpo_distill_single \
	--num_question_generations 100 \
	--num_questions_per_generation 10 \
	--report_student_performance \
	--eval_questions_path ./data/wiki_20/single_passage_questions.jsonl \
	--eval_judge_backend hf \
	--eval_judge_model gpt-5-mini \
	${DEBUG:+--debug} \
	> >(tee $LOG_DIR/main_$(date +%Y%m%d_%H%M%S).out) 2>&1