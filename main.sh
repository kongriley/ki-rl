LOG_DIR="./out/"
DEBUG=

# Usage: ./main.sh   or   MODE=no_gen ./main.sh
MODE="${MODE:-default}"

source .venv/bin/activate

common_args=(
	--dataset_path ./data/wiki_20/data.json
	--num_generation_iterations 50
	--num_question_generations 5
	--num_questions_per_generation 5
	--report_student_performance
	--eval_questions_path ./data/wiki_20/gpt-5-mini_questions.jsonl
	--eval_judge_backend openai
	--eval_judge_model gpt-5-mini
	--save_student_result_copy_dir ./results
	--use_good_questions
	--accumulate_questions
)

case "$MODE" in
default)
	mode_args=(
		--output_dir ./distill/out/grpo_distill
		--save_student_result_copy_name student_model
	)
	;;
no_gen)
	mode_args=(
		--skip_generator_training
		--accumulate_questions
		--output_dir ./distill/out/distill_no_gen
		--save_student_result_copy_name no_gen
	)
	;;
*)
	echo "Unknown MODE='$MODE' (use default or no_gen)" >&2
	exit 1
	;;
esac

[[ -n "${DEBUG:-}" ]] && mode_args+=(--debug)

log="$LOG_DIR/main_$(date +%Y%m%d_%H%M%S).out"
python -u distill/main.py "${common_args[@]}" "${mode_args[@]}" \
	> >(tee "$log") 2>&1
