set -x 

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
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 4 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 4 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 4 \
   --vllm_tensor_parallel_size 2 \
   --colocate_actor_ref \
   --pretrain Qwen/Qwen2.5-Coder-7B-Instruct \
   --save_path /mnt/data/openrlhf/Qwen2.5-Coder-7B-Instruct-code-context-ppo-outputs \
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
   --init_kl_coef 0.001 \
   --prompt_data /mnt/data/openrlhf/code_contests.train.Phi35-mini-ins.topp1.temp1.n4.jsonl \
   --input_key description \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
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


