# !/bin/bash

MODELS=(
    # "meta-llama/Meta-Llama-3-8B"
    # "meta-llama/Meta-Llama-3-8B-Instruct"
    # "meta-llama/Llama-3.2-3B"
    # "meta-llama/Llama-3.2-3B-Instruct"
    Qwen/Qwen2.5-1.5B-Instruct
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

# RUNNAME="initialpolicy-prompt@r1_llama-v0"
# RUNNAME="debuginitialpolicy"
# RUNNAME="initialpolicy-prompt@r1_llama-v0"

RUNNAME="qweninstruct_sample"
# MODEL="meta-llama/Meta-Llama-3.2-3B"
# DATA="./raw_data/math_train_balanced-200"
# DATA="./raw_data/gsm8k_train"
DATA="./raw_data/countdown_train"

# PROMPT="templates/r1_default_llama.txt"
# PROMPT="templates/r1_llama-v0.txt"
# PROMPT="templates/r1_llama_instruct_v0.py"
PROMPT="./templates/qwen_tinyzero_countdown_v0.py"

CUDA_VISIBLE_DEVICES=2 python tests/sample_many.py \
    --run_name $RUNNAME-1.5B-Instruct \
    --model_name ${MODELS[0]} \
    --data_path $DATA \
    --prompt_file $PROMPT \
    --temperature 1.0 \
    --top_p 1.0 \
    --n 8 \
    --max_tokens 2048 \
    --reward_fn countdown \
    --max_sample 100 \
    --input_key problem \
    --input_key target
    # &
# 
# CUDA_VISIBLE_DEVICES=3 python tests/sample_many.py \
#     --run_name $RUNNAME-8B-Instruct \
#     --model_name ${MODELS[1]} \
#     --data_path $DATA \
#     --prompt_file $PROMPT \
#     --temperature 1.0 \
#     --top_p 1.0 \
#     --n 8 \
#     --max_tokens 2048 &

# for i in "${!MODELS[@]}"; do
#     MODEL=${MODELS[i]}
#     echo "Processing model $((i+1))/${#MODELS[@]}: ${MODELS[i]}"
#     CUDA_VISIBLE_DEVICES=${i} python tests/sample_many.py \
#         --run_name $RUNNAME  \
#         --model_name $MODEL \
#         --data_path $DATA  \
#         --prompt_file $PROMPT \
#         --temperature 1.0 \
#         --top_p 1.0 \
#         --n 8 \
#         --max_tokens 2048 &
#         # > /dev/null 2>&1 &
# done


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