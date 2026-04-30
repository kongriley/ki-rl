# Knowledge Injection for Question Generation

## Overview

This script runs the full knowledge injection pipeline.
```bash
./main.sh
```

The repo is organized as follows:
- `distill`
    - `main.py` is the main pipeline script.
    - `distill_phase_worker.py` runs the distillation phase in a subprocess.
    - `grpo_phase_worker.py` runs the GRPO phase in a subprocess.
    - `distil_config.py` and `distil_trainer.py` are pre-existing distillation code.
- `data`
    - `generate_questions.py` generates questions from data in JSON format.
- `eval`
    - `eval_questions.py` evaluates the trained models on a set of questions. This can also run the baseline approaches (closed-book base evaluation, ICL evaluation, RAG evaluation).
    - `plot_student_results.py` plots the results of the trained models.
- `baselines`
    - `cpt.py` and `cpt.sh` run continued pretraining (next-token completion) on the passages. `sanity_check_cpt.py` makes sure the CPT model is actually training to the passages (that is, the perplexity of the CPT model is lower than the perplexity of the base model).
    - `active_reading.py` runs active reading on the passages. 

## Hyperparameters

The training loop runs for `num_generation_iterations` iterations.
1. Question model training:
    - Train the question model for `num_question_model_train_epochs` generations. The number of gradient accumulation steps is `gradient_accumulation_steps` (default 32). The number of GRPO generations sampled per prompt is `num_grpo_generations` (default 4).
    - Question model training is run in `grpo_phase_worker.py`. The reward function is defined below.
    - Save good questions collected from this generator training iteration, if `use_good_questions` is set.
    - Generate remaining questions for each passage in the dataset, with `num_questions_per_generation` questions per prompt generation, until the number of good questions is at least `num_question_generations`.
2. Student model training:
    - Train the student model for `num_train_epochs` generations. The number of gradient accumulation steps is `gradient_accumulation_steps` (default 32). Student model training is run in `distill_phase_worker.py`.
    - If `report_student_performance` is set, evaluate the student model on the generated questions. This uses the `eval_questions_path` (for pre-existing questions) or `eval_question_model` (for generated questions) and `eval_judge_model` to evaluate the student model.

## Reward function

The generated questions are evaluated using the student model with and without the passage as context. 

- If the question is answered correctly without context, the question is too easy and the reward is -0.5.
- If the question is not answered correctly without context, but is answered correctly with context, the question is good and the reward is 1.0.
- Otherwise, if the question is wrong with or without context, the question is garbage (too difficult) and the reward is -1.0.

There are some edge case rewards:
- Sometimes the model will generate a question that is not in the expected format. In this case, the question is penalized by -2.0.
- As a stopgap, sometimes the model will generate a PRESIDIO_ANONYMIZED token in the response in place of a proper noun. In this case, the question is penalized by -1.5.

## Data format
- 2025_disasters: articles of 2025 natural disasters truncated to 2,860 tokens: 2025 Myanmar earthquake, 2025 Kamchatka earthquake, 2025 Uttarakhand flash flood, Typhoon Kalmaegi, Tropical Storm Wipha, Cyclone Ditwah, Hurricane Melissa, 2025 Kentwood-Carson tornado, July 2025 Central Texas floods
- wiki_20: 20 Wikipedia articles sourced from wikimedia/wikipedia