script_dir=$(cd "$(dirname "$0")" && pwd)

nnode1=0.0.0.0
nnode2=0.0.0.1

#检查 GPU
echo 'GPU状态'
pdsh -R ssh -w  $nnode1,$nnode2 'nvidia-smi |grep %'

#检查 GPU
echo '初始化RL环境'
pdsh -R ssh -w  $nnode1,$nnode2 "bash $script_dir/restart_docker_rl.sh"


echo '启动header on node1'
pdsh -R ssh -w  $nnode1 "docker exec rl bash $script_dir/run_head.sh"

echo 'wait ray header launch...'
sleep 10

echo '启动 worker'
pdsh -R ssh -w $nnode2 "docker exec rl bash $script_dir/run_worker.sh"
