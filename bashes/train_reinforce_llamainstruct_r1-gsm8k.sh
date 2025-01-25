set -x

echo $PWD

# model_name="meta-llama/Meta-Llama-3-8B-Instruct"
model_name="meta-llama/Meta-Llama-3-8B-Instruct"
PROMPT="./templates/r1_llama-v0.txt"
PROMPT="./templates/r1_llama_instruct_v0.py"
DATA="./raw_data/gsm8k_train"
SAVE_PATH="./checkpoint/llama-3-8b-instruct_gsm8k_reinforce-r1"

# Debug
read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain $model_name \
   --save_path $SAVE_PATH \
   --ckpt_path $SAVE_PATH-ckpts \
   --num_episodes 2 \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 4 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 128 \
   --n_samples_per_prompt 4 \
   --max_epochs 1 \
   --prompt_max_len 256 \
   --generate_max_len 2048 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --input_template_file $PROMPT \
   --apply_chat_template \
   --answer_key gt_answer \
   --prompt_data $DATA \
   --input_key problem \
   --max_samples 100000 \
   --normalize_reward \
   --adam_offload \
   --advantage_estimator reinforce \
   --flash_attn \
   --load_checkpoint \
   --use_verifiable_reward \
   --gradient_checkpointing \
   --use_wandb 109ad64167d64d59f27db38e751574efa73def3c \
   --save_hf_ckpt \
   --wandb_group trial-runs
EOF

if [[ ${1} != "slurm" ]]; then
    # deepspeed --include localhost:0 --module $training_commands
    deepspeed --master_port 52038 --include localhost:2,3 --module $training_commands
fi


#    --stop_strings "\n\n" \

# Debug
# read -r -d '' training_commands <<EOF
# openrlhf.cli.train_ppo \
#    --pretrain $model_name \
#    --save_path ./checkpoint/llama-3-8b-instruct_reinforce-r1 \
#    --save_steps -1 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --micro_train_batch_size 1 \
#    --train_batch_size 4 \
#    --micro_rollout_batch_size 2 \
#    --rollout_batch_size 8 \
#    --max_epochs 1 \
#    --prompt_max_len 256 \
#    --generate_max_len 2048 \
#    --stop_strings "\n\n" \
#    --zero_stage 2 \
#    --bf16 \
#    --actor_learning_rate 5e-7 \
#    --critic_learning_rate 9e-6 \
#    --init_kl_coef 0.01 \
#    --input_template_file $PROMPT \
#    --answer_key gt_answer\
#    --prompt_data ./raw_data/math_train_balanced-200 \
#    --input_key problem \
#    --max_samples 100000 \
#    --normalize_reward \
#    --adam_offload \
#    --advantage_estimator reinforce \
#    --flash_attn \
#    --load_checkpoint \
#    --use_verifiable_reward \
#    --gradient_checkpointing \
#    --use_wandb 109ad64167d64d59f27db38e751574efa73def3c \
#    --wandb_group trial-runs
# EOF


#    --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \

# input_template_file="templates/r1_default_llama.txt"
# Full
# read -r -d '' training_commands <<EOF
# openrlhf.cli.train_ppo \
#    --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
#    --save_path ./checkpoint/llama-3-8b-rlhf \
#    --save_steps -1 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --micro_train_batch_size 2 \
#    --train_batch_size 16 \
#    --micro_rollout_batch_size 4 \
#    --rollout_batch_size 128 \
#    --max_epochs 1 \
#    --prompt_max_len 256 \
#    --generate_max_len 2048 \
#    --zero_stage 2 \
#    --bf16 \
#    --actor_learning_rate 5e-7 \
#    --critic_learning_rate 9e-6 \
#    --init_kl_coef 0.01 \
#    --input_template_file ${input_template_file} \
#    --answer_key gt_answer\
#    --prompt_data ./raw_data/math_train_balanced-200 \
#    --input_key problem \
#    --max_samples 100000 \
#    --normalize_reward \
#    --adam_offload \
#    --advantage_estimator reinforce \
#    --flash_attn \
#    --load_checkpoint \
#    --use_verifiable_reward \
#    --gradient_checkpointing
# EOF


# # Debug
# read -r -d '' training_commands <<EOF
# openrlhf.cli.train_ppo \
#    --pretrain  \
#    --save_path ./checkpoint/llama-3-8b-rlhf \
#    --save_steps -1 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --micro_train_batch_size 1 \
#    --train_batch_size 4 \
#    --micro_rollout_batch_size 2 \
#    --rollout_batch_size 8 \
#    --max_epochs 1 \
#    --prompt_max_len 256 \
#    --generate_max_len 2048 \
#    --zero_stage 2 \
#    --bf16 \
#    --actor_learning_rate 5e-7 \
#    --critic_learning_rate 9e-6 \
#    --init_kl_coef 0.01 \
#    --input_template_file ${input_template_file} \
#    --answer_key gt_answer\
#    --prompt_data ./raw_data/math_train_balanced-200 \
#    --input_key problem \
#    --max_samples 100000 \
#    --normalize_reward \
#    --adam_offload \
#    --advantage_estimator reinforce \
#    --flash_attn \
#    --load_checkpoint \
#    --use_verifiable_reward \
#    --gradient_checkpointing
# EOF

#    e-apply_chat_template \
   # --packing_samples
    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward
# set -x
#    # --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
# # OpenRLHF/Llama-3-8b-sft-mixture
# # export CUDA_VISIBLE_DEVICES=5

# read -r -d '' training_commands <<EOF
# openrlhf.cli.train_ppo \
#    --pretrain Qwen/Qwen2-0.5B-Instruct \
#    --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
#    --save_path ./checkpoint/llama-3-8b-rlhf \
#    --save_steps -1 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --micro_train_batch_size 2 \
#    --train_batch_size 16 \
#    --micro_rollout_batch_size 4 \
#    --rollout_batch_size 16 \
#    --max_epochs 1 \
#    --prompt_max_len 1024 \
#    --advantage_estimator reinforce \
#    --generate_max_len 1024 \
#    --zero_stage 2 \
#    --bf16 \
#    --actor_learning_rate 5e-7 \
#    --critic_learning_rate 9e-6 \
#    --init_kl_coef 0.01 \
#    --prompt_data OpenRLHF/prompt-collection-v0.1 \
#    --input_key context_messages \
#    --apply_chat_template \
#    --max_samples 100000 \
#    --normalize_reward \
#    --adam_offload \
#    --flash_attn \
#    --load_checkpoint \
#    --gradient_checkpointing
# EOF

#     # --packing_samples
#     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
#     # --remote_rm_url http://localhost:5000/get_reward

# if [[ ${1} != "slurm" ]]; then
#    deepspeed --include localhost:0 --module $training_commands
# fi
