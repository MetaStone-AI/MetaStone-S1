# nccl settings
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=bond1
export NCCL_IB_GID_INDEX=3
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
export NCCL_PXN_DISABLE=0
export GLOO_SOCKET_IFNAME=bond1
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# Set XFormers backend to avoid CUDA errors
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=f4eeb2f2bcbec56195df62e39b6fee6e8f39e108

# Connect to head node (replace with your head node's address)
ray start --address='xxxx'