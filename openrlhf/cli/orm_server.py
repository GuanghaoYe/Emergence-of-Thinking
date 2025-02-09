import concurrent.futures
import datetime
import inspect
import io
import json
import logging
import multiprocessing
import os
import re
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict, load_dataset
from fastapi import FastAPI, Request
from numpy import isin
from pydantic import BaseModel
from tqdm import tqdm

from evaluation.grader import math_equal
from evaluation.parser import extract_answer
from openrlhf.cli.gpt_reward import grade_cot

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Globals (you may need them to be accessible by each process)
args = None
ground_truth_dict = {}
tokenizer = None

class RequestLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create a new log file for each run with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"requests_{timestamp}.jsonl"
        
        # Ensure the log file exists
        self.log_file.touch()
        
        logger.info(f"Logging requests to: {self.log_file}")

    def log_request(self, request_data: dict, reward: float | List[float], processing_time: float):
        """Log a request with its reward and processing time"""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "request": request_data,
            "reward": reward,
            "processing_time": processing_time,
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


def parse_query(sequence):
    # Parse the query
    global args
    if 'llama' in args.model_name.lower():
        query = sequence.split("Problem: ")[1].split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[0].strip()
        response = sequence.split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1].strip()
    elif 'qwen' in args.model_name.lower():
    # <|im_start|>user\nProblem:{input}<|im_end|>\n<|im_start|>assistant\n
        query = sequence.split("<|im_start|>user\nProblem:")[1].split("<|im_end|>")[0].strip()
        response = sequence.split("<|im_start|>assistant\n")[1].strip()
    elif 'cot' in args.model_name.lower():
        query = sequence.split("Question: ")[1].strip()
        response = sequence.split("Answer: ")[1].strip()
    else:
        raise ValueError("Model name not recognized")
    return query, response


def extract_answer_and_reasoning(text):
    # Define the regex pattern to match the outermost \boxed{}
    pattern = r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    answer = extract_answer(text, data_name="math")
    # If a match is found, return the matched group, otherwise return an empty string
    if match:
        return answer, text[:match.start()]
    elif answer:
        return answer, None
    else:
        return None, None


def _compute_single_reward(seq: str) -> float:
    """
    A helper function that computes the reward for a single sequence.
    This function can be safely called in parallel worker processes.
    """
    global args, ground_truth_dict, tokenizer

    try:
        tic = time.time()
        # Parse query & response
        query, response = parse_query(seq)
        ground_truth = ground_truth_dict[re.sub(r'[^a-zA-Z0-9]', '', query)]

        answer, reasoning = extract_answer_and_reasoning(response)
        
        # Base reward: math_equal
        reward = math_equal(ground_truth, answer, timeout=True) + 0.0

        if "boxed{" not in response:
            # From jana and huseyin
            reward -= 0.5


        # Apply length penalty if specified
        if args.length_penalty != 0.0:
            if reasoning is None:
                reasoning_length = len(tokenizer(response)["input_ids"])
            else:
                reasoning_length = len(tokenizer(reasoning)["input_ids"])
            # Subtract a fraction of the length penalty
            reward = reward - args.length_penalty / reasoning_length

        duration = time.time() - tic
        print(time.ctime(), f"GT Reward = {reward:.3f}; GT Duration: {duration:.2f}s")

        # Optionally use GPT-based reward
        if args.use_gpt:
            tic = time.time()
            aux_reward = grade_cot(query, response, grader_model_name="gpt-4o-mini")
            duration = time.time() - tic
            print(time.ctime(), f"GT Reward = {reward:.3f}, GPT Reward = {aux_reward:.3f}; GPT Duration: {duration:.2f}s")
            
            # If GPT fails, use 0.5 as fallback
            if aux_reward == -1:
                aux_reward = 0.5
            # Combine GPT reward with math reward
            reward = reward * 0.8 + aux_reward * 0.2

    except Exception as e:
        # If any error occurs, default to 0.0
        reward = 0.0

    return reward


def calculate_reward(data: dict) -> float | list[float]:
    """
    Calculate rewards in parallel for each sequence in data["query"].
    """
    global args

    sequence = data.get("query", "")
    if isinstance(sequence, str):
        sequence = [sequence]

    # Use a process pool to compute each reward in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        rewards = list(executor.map(_compute_single_reward, sequence))
    return rewards


app = FastAPI()
request_logger = None

@app.on_event("startup")
async def startup_event():
    global request_logger
    request_logger = RequestLogger(log_dir=args.log_dir)
    logger.info("Request logger initialized")


@app.post("/get_reward")
async def get_reward(data: Request):
    start_time = time.time()
    try:
        data = await data.json()
        reward = calculate_reward(data)
        processing_time = time.time() - start_time
        request_logger.log_request(data, reward, processing_time)
    except Exception as e:
        print(f"Error1: {str(e)}")
        reward = 0.0
    return {"rewards": reward}


if __name__ == '__main__':
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description='Run the Code Contests Reward Model server')
    parser.add_argument('--dataset', type=str, help='Path to the dataset', required=True)
    parser.add_argument('--model_name', type=str, help='model name', required=True)
    parser.add_argument('--log_dir', type=str, help='Directory for request logs', default='logs')
    parser.add_argument('--length_penalty', type=float, help='Length penalty for the model', default=20.0)
    parser.add_argument('--use_gpt', type=float, help='Use GPT or not', default=True)
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Only load tokenizer if length_penalty is non-zero
    if args.length_penalty != 0.0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    dataset = load_dataset(args.dataset)

    # Prepare ground-truth dict
    if isinstance(dataset, DatasetDict):
        for split in dataset:
            for example in dataset[split]:
                query = re.sub(r'[^a-zA-Z0-9]', '', example['problem'])
                ground_truth_dict[query] = example['answer']
    elif isinstance(dataset, Dataset):
        for example in dataset:
            query = re.sub(r'[^a-zA-Z0-9]', '', example['problem'])
            ground_truth_dict[query] = example['answer']
    else:
        raise ValueError("Dataset format not recognized")

    uvicorn.run(app, host="0.0.0.0", port=8000)
