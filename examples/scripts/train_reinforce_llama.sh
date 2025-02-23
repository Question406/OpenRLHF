set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain OpenRLHF/Llama-3-8b-sft-mixture \
   --reward_pretrain OpenRLHF/Llama-3-8b-rm-mixture \
   --save_path ./checkpoint/llama-3-8b-rlhf \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 8 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 32 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --max_samples 100000 \
   --normalize_reward \
   --adam_offload \
   --advantage_estimator reinforce \
   --flash_attn \
   --load_checkpoint \
   --gradient_checkpointing
EOF

   # --packing_samples
    # --packing_samples
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    deepspeed --include localhost:0 --module $training_commands
fi

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
