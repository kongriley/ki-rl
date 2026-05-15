
# Usage: ./main.sh   or   MODE=no_gen ./main.sh
MODE="${MODE:-default}"

DATASET="wiki_20"
LOG_DIR="./out/logs/"
DEBUG=

NUM_ITERATIONS=3

source .venv/bin/activate

common_args=(
	--dataset_path ./data/${DATASET}/data.json
	--num_generation_iterations ${NUM_ITERATIONS}
	--num_question_generations 5
	--num_questions_per_generation 5
	--report_student_performance
	--eval_questions_path ./data/${DATASET}/gpt-5-mini_questions.jsonl
	--eval_judge_backend openai
	--eval_judge_model gpt-5-mini
	--save_student_result_copy_dir ./results/${DATASET}
	--use_good_questions
	--num_distill_epochs 1 # to test
	--grpo_beta 0.001
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
