set -x

echo $PWD


model_name="Qwen/Qwen2.5-Math-7B"
PROMPT="./templates/qwen_math_v0.py"
DATA="./raw_data/math_train"
SAVE_PATH="./checkpoint/qwen2.5-Math7B_math"

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain $model_name \
   --critic_pretrain $model_name \
   --save_path $SAVE_PATH \
   --ckpt_path $SAVE_PATH-ckpts \
   --num_episodes 20 \
   --save_steps 10 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 512 \
   --n_samples_per_prompt 1 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 3000 \
   --verifiable_reward_fn math_simple \
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
   --advantage_estimator gae \
   --flash_attn \
   --load_checkpoint \
   --use_verifiable_reward \
   --gradient_checkpointing \
   --use_wandb 109ad64167d64d59f27db38e751574efa73def3c \
   --save_hf_ckpt \
   --wandb_group trial-runs-2
EOF

#    --stop_strings "</answer>" \
if [[ ${1} != "slurm" ]]; then
    # deepspeed --master_port 52038 --include localhost:2,3 --module $training_commands
    # deepspeed --master_port 52038 --include localhost:2 --module $training_commands
    deepspeed --master_port 52038 --module $training_commands
fi
#    --stop_strings "</answer>" \
#    --micro_train_batch_size 8 \
#    --train_batch_size 256 \
#    --micro_rollout_batch_size 16 \
#    --rollout_batch_size 256 \