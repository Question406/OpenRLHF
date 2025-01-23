# !/bin/bash

MODELS=(
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Meta-Llama-3.2-3B"
    "meta-llama/Meta-Llama-3.2-3B-Instruct"
)
TEMPERATURES=(
    0.5
    0.7
    1.0
)

TOPPs=(
    0.5
    0.7
    1.0
)

RUNNAME="initialpolicy"

MODEL="meta-llama/Meta-Llama-3.2-3B"

DATA="raw_data/math_train_balanced-200"

PROMPT="templates/r1_default_llama.txt"

for MODEL in "${MODELS[@]}"; do
    python tests/sample_many.py \
        --run_name $RUNNAME  \
        --model_name $MODEL \
        --data_path $DATA  \
        --prompt_file $PROMPT \
        --temperature 1 \
        --top_p 1 \
        --n 8 \
        --max_tokens 2048
done

# python tests/sample_many.py \
#     --run_name initialpolicy  \
#     --model_name $MODEL \
#     --data_path $DATA  \
#     --prompt_file $PROMPT \
#     --temperature 1 \
#     --top_p 1 \
#     --n 8 \
#     --max_tokens 2048 \
