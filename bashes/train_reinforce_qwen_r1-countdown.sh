set -x

echo $PWD

model_name="Qwen/Qwen2.5-1.5B"
PROMPT="./templates/qwen_tinyzero_base_v0.txt"
DATA="./raw_data/countdown_train"
SAVE_PATH="./checkpoint/qwen2.5-1.5B_coutndown"

# read -r -d '' training_commands <<EOF
# openrlhf.cli.train_ppo \
#    --pretrain $model_name \
#    --save_path $SAVE_PATH \
#    --ckpt_path $SAVE_PATH-ckpts \
#    --num_episodes 1 \
#    --save_steps 50 \
#    --logging_steps 1 \
#    --eval_steps -1 \
#    --stop_strings "</answer>" \
#    --micro_train_batch_size 4 \
#    --train_batch_size 8 \
#    --micro_rollout_batch_size 8 \
#    --rollout_batch_size 8 \
#    --n_samples_per_prompt 4 \
#    --max_epochs 1 \
#    --prompt_max_len 256 \
#    --generate_max_len 2048 \
#    --zero_stage 2 256 \
#    --bf16 \
#    --actor_learning_rate 5e-7 \
#    --critic_learning_rate 9e-6 \
#    --init_kl_coef 0.01 \
#    --input_template_file $PROMPT \
#    --verifiable_reward_fn countdown \
#    --answer_key gt_answer \
#    --prompt_data $DATA \
#    --input_key problem target \
#    --max_samples 100000 \
#    --normalize_reward \
#    --adam_offload \
#    --advantage_estimator reinforce \
#    --flash_attn \
#    --load_checkpoint \
#    --use_verifiable_reward \
#    --gradient_checkpointing \
#    --use_wandb 109ad64167d64d59f27db38e751574efa73def3c \
#    --save_hf_ckpt \
#    --wandb_group trial-runs
# EOF

# if [[ ${1} != "slurm" ]]; then
#     # deepspeed --include localhost:0 --module $training_commands
#     # deepspeed --include localhost:4,5,6,7 --module $training_commands
#     deepspeed --include localhost:1 --module $training_commands
# fi

# real
read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain $model_name \
   --save_path $SAVE_PATH \
   --ckpt_path $SAVE_PATH-ckpts \
   --num_episodes 1 \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps -1 \
   --stop_strings "</answer>" \
   --micro_train_batch_size 8 \
   --train_batch_size 256 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 256 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 256 \
   --generate_max_len 2048 \
   --verifiable_reward_fn countdown \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --input_template_file $PROMPT \
   --answer_key gt_answer \
   --prompt_data $DATA \
   --input_key problem target \
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
    deepspeed --include localhost:0,1 --module $training_commands
fi


