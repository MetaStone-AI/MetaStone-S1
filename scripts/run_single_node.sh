#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
export WANDB_MODE=offline

data_path="./data"

model_path="agentica-org/DeepScaleR-1.5B-Preview"
score_model_path="null"
output_dir="./output"

base_name="S1-1.5B"
exp_name="$base_name"

lr=1e-6
train_batch_size=32
ppo_mini_batch_size=8
ppo_micro_batch_size=4
val_batch_size=1
max_prompt_length=1024
max_response_length=8192
n_gpus_per_node=8

ppo_max_token_len_per_gpu=$(((max_prompt_length + max_response_length)))

rollout_n=8
kl_coef=0.001


python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$data_path/train.parquet \
    data.val_files=$data_path/aime.parquet \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    actor_rollout_ref.model.path=$model_path  \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$ppo_micro_batch_size \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=$kl_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.use_score=True \
    actor_rollout_ref.score.model_path=$score_model_path \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=$output_dir \
    trainer.default_local_dir=$output_dir \
    trainer.project_name="$base_name" \
    trainer.experiment_name="$exp_name" \
    +trainer.val_before_train=False \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    trainer.total_epochs=30 "${@:1}"
