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
    ray start --head --node-ip-address node-0 --num-gpus 8

   for i in $(echo $AZ_BATCH_NODE_LIST | tr ";" "\n")
    do
        if [ "$i" = "node-0" ]; then
            continue
        fi
        ssh $i "export PATH=\$HOME/.local/bin/:\$PATH ; ray start --address node-0:6379 --num-gpus 8"
    done

    sleep 10s

   yes | ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{"working_dir": "/scratch/amlt_code"}' \
   -- python3 -m openrlhf.cli.train_ppo_ray \
    $node_setup \
   --colocate_actor_ref \
   --pretrain /mnt/data/hfmodels/Llama-3.1-8B-Instruct \
   --save_path /mnt/data/openrlhf/Llama-3.1-8B-Instruct-code-context-ppo-outputs \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 32 \
   --rollout_batch_size 128 \
   --max_samples 100000 \
   --max_epochs 1 \
   --num_episodes 50 \
   --prompt_max_len 2048 \
   --generate_max_len 2048 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.0 \
   --save_steps 40 \
   --prompt_data /mnt/data/openrlhf/code_contests.train.Phi35-mini-ins.topp1.temp1.n4.jsonl \
   --input_key description \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --note "Pub+Gen" \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --vllm_sync_backend gloo \
   --use_wandb $WANDB_API_KEY \
   --remote_rm_url http://node-0:8000/get_reward \

   # --runtime-env-json='{"setup_commands": ["pip install openrlhf[vllm]"]}' [Install deps]
   # --ref_reward_offload [Offload to CPU]
   # --remote_rm_url http://localhost:5000/get_reward

else
    # Start the Ray worker node
    echo "Starting Ray worker node"
fi


