set -x
 
NUM_NODE=$1
 
echo $AZ_BATCH_NODE_LIST
echo $MASTER_ADDR
 
if [ "$NUM_NODE" -eq 2 ]; then
  node_setup="--ref_num_nodes 1 \
  --ref_num_gpus_per_node 4 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 8 \
  --vllm_num_engines 2 \
  --vllm_tensor_parallel_size 2"
elif [ "$NUM_NODE" -eq 1 ]; then
  node_setup="--ref_num_nodes 1 \
  --ref_num_gpus_per_node 2 \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 4 \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 2"
else
  echo "Unsupported NUM_NODE value: $NUM_NODE"
  exit 1
fi
 
if [ "$NODE_RANK" -eq 0 ]; then
    # Start the Ray head node
    ray start --head --node-ip-address $MASTER_ADDR --num-gpus 8
 
  #  for i in $(echo $AZ_BATCH_NODE_LIST | tr ";" "\n")
  #   do
  #       if [ "$i" = "node-0" ]; then
  #           continue
  #       fi
  #       ssh $i "export PATH=\$HOME/.local/bin/:\$PATH ; ray start --address node-0:6379 --num-gpus 8"
  #   done
 
    sleep 10s
 
   yes | ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/tmp/amlt_code"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
    $node_setup \
   --colocate_actor_ref \
   --pretrain /mnt/data/huinan/openrlhf/sft/checkpoint3 \
   --save_path /mnt/data/openrlhf/ppo/checkpointcreaterewardreinforce \
   --micro_train_batch_size 8 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 16 \
   --rollout_batch_size 512 \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes 500 \
   --prompt_max_len 2048 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-7 \
   --critic_learning_rate 2e-6 \
   --init_kl_coef 0.05 \
   --save_steps 40 \
   --prompt_data /mnt/data/huinan/openrlhf/math_formatted_sft_only_incorrect_by_sft_ckpt5_greedy_no_asy.jsonl \
   --input_key input \
   --normalize_reward \
   --packing_samples \
   --note "PPO_reinforce" \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --vllm_sync_backend nccl \
   --use_wandb $WANDB_API_KEY \
   --remote_rm_url http://$MASTER_ADDR:8000/get_reward \
   --temperature 0.7 \
   --max_ckpt_num 10 \
   --l2 0.1 \
   --ckpt_path /mnt/data/openrlhf/ppo/checkpointcreaterewardreinforce \
   --advantage_estimator reinforce \
   --overlap_comm  \
   
   # --ckpt_path /mnt/data/huinan/openrlhf/ppo/checkpoint9 \
   # --n_samples_per_prompt 10 \
   # --adam_offload \
 
 
   # --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
   # --ref_reward_offload [Offload to CPU]
   # --remote_rm_url http://localhost:5000/get_reward
 
else
    sleep 20s
    ray start --address $MASTER_ADDR:6379 --num-gpus 8
    # Start the Ray worker node
    echo "Starting Ray worker node"
fi
 