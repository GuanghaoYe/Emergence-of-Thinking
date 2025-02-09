set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset AtAndDev/QwQ-LongCoT-59k-cleaned \
   --input_key problem \
   --output_key qwq \
   --train_batch_size 24 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain /mnt/data/hfmodels/Qwen2.5-Coder-32B-Instruct/ \
   --save_path /mnt/data/outputs/Qwen2.5-Coder-32B-Instruct-QwQ-LongCoT-59k-cleaned-max_samples2048 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing 
EOF
    # --packing_samples


if [ "$NODE_RANK" -eq 0 ]; then
    deepspeed --module $training_commands
fi


# --dataset AtAndDev/QwQ-LongCoT-59k-cleaned \



read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8192 \
   --dataset /mnt/data/outputs/QwQ-LongCoT-cleaned-32k-math-only.jsonl \
   --input_key problem \
   --output_key qwq \
   --train_batch_size 24 \
   --micro_train_batch_size 1 \
   --max_samples 2048 \
   --pretrain /mnt/data/hfmodels/Qwen2.5-Coder-32B-Instruct/ \
   --save_path /mnt/data/outputs/Qwen2.5-Coder-32B-Instruct-QwQ-LongCoT-cleaned-32k-math-only-max_sample2048 \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 2 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing 
EOF
    # --packing_samples


if [ "$NODE_RANK" -eq 0 ]; then
    deepspeed --module $training_commands
fi