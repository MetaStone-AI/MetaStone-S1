#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1

data_path="./data"

model_path="Qwen/QWQ-32B"
resume_model_path="/path/to/MetaStone-L1-32B"
output_dir="./output"
mkdir -p $output_dir

n_nodes=16
n_gpus_per_node=8
rollout_tp_size=8 

base_name="S1-32B"
exp_name="$base_name"

max_prompt_length=2048
max_response_length=32768
ppo_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length)*2 ))
train_batch_size=128
ppo_mini_batch_size=256
ulysses_sequence_parallel_size=8

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path/train.parquet \
    data.val_files=$data_path/aime.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.shuffle=False \
    data.resume=True \
    data.resume_global_steps=1 \
    reward_model.reward_manager=prime \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.use_score=True \
    actor_rollout_ref.score.channel=5120 \
    actor_rollout_ref.score.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
    actor_rollout_ref.actor.optim.no_load_optim=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp_size \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    +actor_rollout_ref.rollout.swap_space=256 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    actor_rollout_ref.actor.optim.no_load_optim=True \
    trainer.resume_mode=$resume_model_path \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_gsm8k' \
    trainer.default_hdfs_dir=$output_dir \
    trainer.default_local_dir=$output_dir \
    trainer.project_name="$base_name" \
    trainer.experiment_name="$exp_name" \
    trainer.experiment_name='qwen2_7b_function_rm' \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$n_nodes \
    trainer.save_freq=5 \
    trainer.test_freq=1000000 \
    trainer.total_epochs=1 2>&1 | tee $output_dir/train.log


