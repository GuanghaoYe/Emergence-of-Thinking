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
else
  echo "Unsupported NUM_NODE value: $NUM_NODE"
  exit 1
fi

if [ "$NODE_RANK" -eq 0 ]; then
    # Start the Ray head node
    ray start --head --node-ip-address 127.0.0.1 --num-gpus 8

   for i in $(echo $AZ_BATCH_NODE_LIST | tr ";" "\n")
    do
        if [ "$i" = "node-0" ]; then
            continue
        fi
        ssh $i "export PATH=\$HOME/.local/bin/:\$PATH ; ray start --address node-0:6379 --num-gpus 8"
    done

    sleep 10s

   yes | ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/tmp/amlt_code"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
    $node_setup \
   --colocate_actor_ref \
   --pretrain /mnt/data/huinan/openrlhf/sft/checkpoint3 \
   --save_path /mnt/data/openrlhf/ppo/checkpoint13 \
   --micro_train_batch_size 1 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 8 \
   --rollout_batch_size 512 \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes 100 \
   --prompt_max_len 2048 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 2e-7 \
   --critic_learning_rate 2e-6 \
   --init_kl_coef 0.05 \
   --save_steps 20 \
   --prompt_data /mnt/data/huinan/openrlhf/math_formatted_sft_only_incorrect_by_sft_ckpt5_greedy_no_asy.jsonl \
   --input_key input \
   --normalize_reward \
   --packing_samples \
   --note "Pub+Gen" \
   --flash_attn \
   --gradient_checkpointing \
   --freezing_actor_steps 10 \
   --load_checkpoint \
   --vllm_sync_backend gloo \
   --use_wandb $WANDB_API_KEY \
   --remote_rm_url http://127.0.0.1:8000/get_reward \
   --adam_offload \
   --temperature 0.7 \
   --max_ckpt_num 10 \


   # --ckpt_path /mnt/data/huinan/openrlhf/ppo/checkpoint9 \
   # --n_samples_per_prompt 10 \
   # --adam_offload \


   # --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
   # --ref_reward_offload [Offload to CPU]
   # --remote_rm_url http://localhost:5000/get_reward

else
    # Start the Ray worker node
    echo "Starting Ray worker node"
fi


