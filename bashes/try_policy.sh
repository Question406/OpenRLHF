# !/bin/bash

MODELS=(
    "meta-llama/Meta-Llama-3-8B"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "meta-llama/Llama-3.2-3B"
    "meta-llama/Llama-3.2-3B-Instruct"
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
PROMPT="templates/r1_llama-v0.txt"


for i in "${!MODELS[@]}"; do
    MODEL=${MODELS[i]}
    echo "Processing model $((i+1))/${#MODELS[@]}: ${MODELS[i]}"
    CUDA_VISIBLE_DEVICES=${i} python tests/sample_many.py \
        --run_name $RUNNAME  \
        --model_name $MODEL \
        --data_path $DATA  \
        --prompt_file $PROMPT \
        --temperature 1.0 \
        --top_p 1.0 \
        --n 8 \
        --max_tokens 2048 > /dev/null 2>&1 &
done


# Run single
# python tests/sample_many.py \
#         --run_name $RUNNAME  \
#         --model_name $MODEL \
#         --data_path $DATA  \
#         --prompt_file $PROMPT \
#         --temperature 1 \
#         --top_p 1 \
#         --n 8 \
#         --max_tokens 2048 