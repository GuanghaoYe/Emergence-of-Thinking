set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 8196 \
   --dataset AtAndDev/QwQ-LongCoT-59k-cleaned \
   --input_key problem \
   --output_key qwq \
   --train_batch_size 24 \
   --micro_train_batch_size 1 \
   --max_samples 500000 \
   --pretrain /mnt/data/hfmodels/Qwen2.5-Coder-32B-Instruct/ \
   --save_path /mnt/data/outputs/Qwen2.5-Coder-32B-Instruct-QwQ-LongCoT-59k-cleaned \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY
EOF
    # --packing_samples


if [ "$NODE_RANK" -eq 0 ]; then
    deepspeed --module $training_commands
fi