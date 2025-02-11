# Emergence of Thinking

This repository contains the code for the paper "On the Emergence of Thinking in LLMs I: Searching for the Right Intuition"

[arXiv link](https://arxiv.org/abs/2502.06773)

## Environment

```bash
bash create_env.sh
pip install -e .
```

## Figure 5(a) of the paper

```bash
python -m openrlhf.cli.orm_server_efficient --dataset evaluation/data/math --model_name meta-llama/Llama-3.1-8B-Instruct --log_dir ./logs/openrlhf_train_ppo --length_penalty 0.0 --use_gpt 0 &
mkdir -p /tmp/code &
bash train_ppo_llama_ray_8B_rm_multi.sh 3 
```

## Figure 5(b) of the paper
```bash
python -m openrlhf.cli.orm_server_efficient --dataset evaluation/data/math --model_name meta-llama/Llama-3.1-8B-Instruct --log_dir ./logs/openrlhf_train_ppo --length_penalty 1000 --use_gpt 0 &
mkdir -p /tmp/code &
bash train_ppo_llama_ray_8B_rm_multi.sh 3 
```

## Evaluation after the PPO training
```bash
python -m evaluation.eval_math_data_parallel --config ./eval_config.yaml
```

## Qwen2.5-32B-Instruct experiment
```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-32B-Instruct
python -m openrlhf.cli.orm_server_efficient --dataset evaluation/data/aime_full_except_24 --model_name Qwen/Qwen2.5-32B-Instruct --log_dir ./logs/openrlhf_train_ppo --length_penalty 1000 --use_gpt 1 &
mkdir -p /tmp/code &
bash train_ppo_qwen_ray_32B_rm_multi.sh 3 
```

## Acknowledgement
The repo is based on the code from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) and the evaluation code is taken from [Qwen](https://github.com/QwenLM/Qwen2.5-Math)

