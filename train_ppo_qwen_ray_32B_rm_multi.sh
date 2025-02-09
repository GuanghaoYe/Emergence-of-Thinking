set -x 

NUM_NODE=$1

if [ "$NUM_NODE" -eq 2 ]; then
  node_setup="--ref_num_nodes 1 \
  --ref_num_gpus_per_node 4 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 4 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --vllm_num_engines 4 \
  --vllm_tensor_parallel_size 2"
elif [ "$NUM_NODE" -eq 1 ]; then
  node_setup="--ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 2 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2"
elif [ "$NUM_NODE" -eq 3 ]; then
  node_setup="--ref_num_nodes 1 \
  --ref_num_gpus_per_node 8 \
  --critic_num_nodes 1 \
  --critic_num_gpus_per_node 8 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 8 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 4"
elif [ "$NUM_NODE" -eq 6 ]; then
  node_setup="--ref_num_nodes 2 \
  --ref_num_gpus_per_node 8 \
  --critic_num_nodes 2 \
  --critic_num_gpus_per_node 8 \
  --actor_num_nodes 2 \
  --actor_num_gpus_per_node 8 \
  --vllm_num_engines 4 \
  --vllm_tensor_parallel_size 4"
else
  echo "Unsupported NUM_NODE value: $NUM_NODE"
  exit 1
fi

if [ "$NODE_RANK" -eq 0 ]; then
    # Start the Ray head node
    ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8

    sleep 10s

   yes | ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/tmp/code"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
    $node_setup \
   --colocate_actor_ref \
   --pretrain Qwen/Qwen2.5-32B-Instruct \
   --save_path ./checkpoint \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes 1000 \
   --prompt_max_len 2048 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-7 \
   --critic_learning_rate 2e-6 \
   --init_kl_coef 0.05 \
   --save_steps 20 \
   --prompt_data ./data/aime_formatted_qwen.jsonl \
   --input_key input \
   --normalize_reward \
   --packing_samples \
   --note "Pub+Gen" \
   --flash_attn \
   --gradient_checkpointing \
   --freezing_actor_steps 4 \
   --load_checkpoint \
   --use_wandb $WANDB_API_KEY \
   --remote_rm_url http://$MASTER_ADDR:8000/get_reward \
   --temperature 0.7 \
   --max_ckpt_num 10 \
   --adam_offload \
   --save_hf_ckpt \
   --disable_ds_ckpt \
   --lambd 0.95 \

else
    sleep 20s
    ray start --address $MASTER_ADDR:6379 --num-gpus 8
    # Start the Ray worker node
    echo "Starting Ray worker node"
fi


