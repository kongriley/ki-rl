LOG_DIR="./out/"
DEBUG=

# python -u distill/main.py \
# 	--dataset_path ./data/wiki_20/data.json \
# 	--output_dir ./distill/out/grpo_distill \
# 	--num_generation_iterations 50 \
# 	--num_question_generations 5 \
# 	--num_questions_per_generation 5 \
# 	--report_student_performance \
# 	--eval_questions_path ./data/wiki_20/gpt-5-mini_questions.jsonl \
# 	--eval_judge_backend hf \
# 	--eval_judge_model gpt-5-mini \
# 	${DEBUG:+--debug} \
# 	> >(tee $LOG_DIR/main_$(date +%Y%m%d_%H%M%S).out) 2>&1

python -u distill/main.py \
	--dataset_path ./data/wiki_20/data.json \
	--output_dir ./distill/out/distill_no_gen \
	--num_generation_iterations 50 \
	--num_question_generations 5 \
	--num_questions_per_generation 5 \
	--report_student_performance \
	--skip_generator_training \
	--eval_questions_path ./data/wiki_20/gpt-5-mini_questions.jsonl \
	--eval_judge_backend hf \
	--eval_judge_model gpt-5-mini \
	${DEBUG:+--debug} \
	> >(tee $LOG_DIR/main_$(date +%Y%m%d_%H%M%S).out) 2>&1