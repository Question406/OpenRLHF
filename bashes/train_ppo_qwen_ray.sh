set -x 

# ray start --head --node-ip-address 0.0.0.0 --num-gpus 8

WANDBKEY="109ad64167d64d59f27db38e751574efa73def3c"
MODEL="Qwen/Qwen2.5-Math-7B"
DATA="./raw_data/jxhe_train"
PROMPT="./templates/qwen_math_v0.py"
SAVE_PATH="./checkpoint/qwen2.5-math7b-math-MathJxhe"

   # --runtime-env-json='{"working_dir": "./openrlhf"}' \

# debug 
ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --pretrain $MODEL \
   --critic_pretrain $MODEL \
   --colocate_actor_ref \
   --save_path $SAVE_PATH \
   --ckpt_path $SAVE_PATH-ckpts \
   --micro_train_batch_size 2 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 1024 \
   --temperature 0.6 \
   --n_samples_per_prompt 4 \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes 20 \
   --prompt_max_len 1024 \
   --generate_max_len 3000 \
   --zero_stage 3 \
   --save_steps 5 \
   --logging_steps 1 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data $DATA \
   --use_verifiable_reward \
   --verifiable_reward_fn math_simple \
   --input_template_file $PROMPT \
   --apply_chat_template \
   --input_key problem \
   --answer_key gt_answer \
   --normalize_reward \
   --adam_offload \
   --advantage_estimator gae \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --wandb_group jxheprompt \
   --use_wandb $WANDBKEY \
   --wandb_project ray_trialrun_ppo

   # --packing_samples \
# Full
# ray job submit --address="http://127.0.0.1:8265" \
#    -- python3 -m openrlhf.cli.train_ppo_ray \
#    --ref_num_nodes 1 \
#    --ref_num_gpus_per_node 4 \
#    --critic_num_nodes 1 \
#    --critic_num_gpus_per_node 2 \
#    --actor_num_nodes 1 \
#    --actor_num_gpus_per_node 4 \
#    --vllm_num_engines 1 \
#    --vllm_tensor_parallel_size 2 \
#    --pretrain $MODEL \
#    --critic_pretrain $MODEL \
#    --colocate_actor_ref \
#    --save_path ./checkpoint/llama3-8b-rlhf \
#    --micro_train_batch_size 8 \
#    --train_batch_size 128 \
#    --micro_rollout_batch_size 32 \
#    --rollout_batch_size 1024 \
#    --max_samples 100000 \
#    --max_epochs 1 \
#    --prompt_max_len 1024 \
#    --generate_max_len 3000 \
#    --zero_stage 3 \
#    --bf16 \
#    --actor_learning_rate 5e-7 \
#    --critic_learning_rate 9e-6 \
#    --init_kl_coef 0.01 \
#    --prompt_data $DATA \
#    --use_verifiable_reward \
#    --verifiable_reward_fn math_simple \
#    --input_template_file $PROMPT \
#    --input_key problem \
#    --answer_key gt_answer \
#    --apply_chat_template \
#    --normalize_reward \
#    --packing_samples \
#    --adam_offload \
#    --advantage_estimator gae \
#    --flash_attn \
#    --gradient_checkpointing \
#    --load_checkpoint \
#    --use_wandb $WANDBKEY

# --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
# --ref_reward_offload [Offload to CPU]
# --remote_rm_url http://localhost:5000/get_reward

# --vllm_sync_backend nccl (Only for multi-nodes with vLLM 0.6.4+ or vLLM 0.4.2)