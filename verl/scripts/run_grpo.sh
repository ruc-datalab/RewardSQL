set -x

export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nproc_per_node=8
CONFIG_PATH="./verl/scripts/grpo_trainer.yaml"

python3 -m verl.trainer.main_ppo \
    --config_path=$CONFIG_PATH