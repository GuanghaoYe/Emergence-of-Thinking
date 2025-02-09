import aiohttp
import asyncio
import time
import random
from datasets import load_dataset
from typing import List, Dict
import statistics
import json

async def send_queries(session: aiohttp.ClientSession, url: str, queries: List[str]) -> tuple[List[float], float]:
    """Send multiple queries and return the rewards and time taken."""
    start_time = time.time()
    
    data = {
        "query": queries  # Send multiple queries as a list
    }
    
    async with session.post(url, json=data) as response:
        result = await response.json()
        end_time = time.time()
        
    return result["rewards"], end_time - start_time

async def run_benchmark(dataset_path: str, num_queries: int = 100, batch_size: int = 10):
    """
    Run benchmark with specified number of queries.
    
    Args:
        dataset_path: Path to the dataset
        num_queries: Number of queries to run
        batch_size: Number of concurrent requests
    """
    # Load dataset
    with open(dataset_path, "r") as f:
        dataset = [json.loads(line) for line in f]
    if isinstance(dataset, dict):
        # If dataset is a DatasetDict, use the first split
        dataset = list(dataset.values())[0]
    
    # Select random queries from dataset
    random.seed(42)
    all_queries = [example['query'].strip() for example in dataset]
    selected_queries = random.sample(all_queries, min(num_queries, len(all_queries)))
    
    url = "http://localhost:8000/get_reward"
    rewards = []
    processing_times = []
    
    async with aiohttp.ClientSession() as session:
        # Process queries in batches
        # i = 464
        # if True:
        for i in range(0, len(selected_queries), batch_size):
            # Send batch of queries in a single request
            batch = selected_queries[i:i+batch_size]
            batch_rewards, batch_time = await send_queries(session, url, batch)
            
            # For multiple queries in one request, we get a list of rewards
            rewards.extend(batch_rewards if isinstance(batch_rewards, list) else [batch_rewards])
            processing_times.append(batch_time)
            if batch_time > 5.0:  # Threshold of 5 seconds
                print(f"\nSlow batch detected (Batch {i//batch_size + 1}):")
                print(f"Time taken: {batch_time:.2f} seconds")

            
            print(f"Processed {min(i + batch_size, len(selected_queries))}/{len(selected_queries)} queries")
    
    # Calculate statistics
    stats = {
        "total_queries": len(processing_times),
        "total_time": sum(processing_times),
        "average_time_per_batch": statistics.mean(processing_times),
        "median_time_per_batch": statistics.median(processing_times),
        "min_time_per_batch": min(processing_times),
        "max_time_per_batch": max(processing_times),
        "std_dev_time_per_batch": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
        "average_reward": statistics.mean(rewards),
        "queries_per_second": len(selected_queries) / sum(processing_times)
    }
    
    # Print results
    print("\nBenchmark Results:")
    print(json.dumps(stats, indent=2))
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark the Reward Model server')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--num_queries', type=int, default=100, help='Number of queries to run')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of concurrent requests')
    
    args = parser.parse_args()
    
    asyncio.run(run_benchmark(
        dataset_path=args.dataset,
        num_queries=args.num_queries,
        batch_size=args.batch_size
    ))