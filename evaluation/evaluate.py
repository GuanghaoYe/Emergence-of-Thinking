import argparse

import numpy as np
from tqdm import tqdm

from evaluation import grader
from evaluation import parser
from evaluation import utils

from collections import Counter, defaultdict

def majority_voting(samples):
    acc = 0
    for sample in samples:
        candidate_answers = Counter()
        candidate_answer_correctness = defaultdict(list)
        for pred, score in zip(sample['pred'], sample['score']):
            candidate_answers[pred] += 1
            candidate_answer_correctness[pred].append(score)
                
        # pick the most common answer
        most_common_answer = candidate_answers.most_common(1)[0][0]
        acc += Counter(candidate_answer_correctness[most_common_answer]).most_common(1)[0][0]
    
    pass_rate = acc / len(samples)

    return pass_rate

# def majority_voting_with_reward(group_by_samples, key):
#     acc = 0
#     for problem, samples in group_by_samples.items():
#         candidate_answer_score = defaultdict(float)
#         candidate_answer_correctness = defaultdict(list)
#         for sample in samples:
#             candidate_answer_score[sample["answer"]] += sample[key]
#             candidate_answer_correctness[sample["answer"]].append(sample['is_correct'])
                
#         # pick the most common answer
#         highest_scored_answer = max(candidate_answer_score, key=candidate_answer_score.get)
#         acc += Counter(candidate_answer_correctness[highest_scored_answer]).most_common(1)[0][0]
    
#     pass_rate = acc / len(group_by_samples)

#     return pass_rate


def evaluate(data_name, samples: list=None, file_path: str=None, max_num_samples=None):
    assert samples or file_path, "samples or file_path must be provided"
    if not samples:
        samples = list(utils.load_jsonl(file_path))
    if 'idx' in samples[0]:
        samples = {sample['idx']: sample for sample in samples}.values()
        samples = sorted(samples, key=lambda x: x['idx']) 
    else:
        samples = [dict(idx=idx, **sample) for idx, sample in enumerate(samples)]

    if max_num_samples:
        print(f"max_num_samples: {max_num_samples} / {len(samples)}")
        samples = samples[:max_num_samples]
    
    # parse gt
    # for sample in samples:
    #     sample['gt_cot'], sample['gt'] = parser.parse_ground_truth(sample, data_name)
    # params = [(idx, pred, sample['gt']) for idx, sample in enumerate(samples) for pred in sample['pred']]

    # scores = []
    # timeout_cnt = 0 

    # for param in params:
    #     result = grader.math_equal_process(param)
    #     scores.append(result)

    # group_by_problems = {}
    # for sample in samples:

    # idx = 0
    # score_mat = []
    # for sample in samples:
    #     sample['score'] = scores[idx: idx+len(sample['pred'])]
    #     assert len(sample['score']) == len(sample['pred'])
    #     score_mat.append(sample['score'])
    #     idx += len(sample['pred'])

    # max_len = max([len(s) for s in score_mat])

    # for i, s in enumerate(score_mat):
    #     if len(s) < max_len:
    #         score_mat[i] = s + [s[-1]] * (max_len - len(s)) # pad

    # # output mean of each column of scores
    # col_means= np.array(score_mat).mean(axis=0)
    # mean_score = list(np.round(col_means * 100, decimals=1))

    for sample in samples:
        sample['gt_cot'], sample['gt'] = parser.parse_ground_truth(sample, data_name)
        sample['score'] = [grader.math_equal(pred, sample['gt']) for pred in sample['pred']]

    result_json = {
        "num_samples": len(samples),
        # "num_scores": len(scores),
        # "timeout_samples": timeout_cnt,
        "empty_samples": len([s for s in samples if not s['pred'][-1]]),
        # "acc": mean_score[0]
        "average_acc": np.mean([np.mean(sample['score']) for sample in samples]),
        "pass_acc": np.mean([np.any(sample['score']) for sample in samples]),
        "maj_acc": majority_voting(samples)
    }

    # each type score
    if "type" in samples[0]:
        type_scores = {}
        for sample in samples:
            if sample['type'] not in type_scores:
                type_scores[sample['type']] = []
            type_scores[sample['type']].append(sample['score'][-1])
        type_scores = {k: np.round(np.array(v).mean() * 100, decimals=1) for k, v in type_scores.items()}
        type_scores = {k: v for k, v in sorted(type_scores.items(), key=lambda item: item[0])}
        result_json['type_acc'] = type_scores

    print(result_json)
    return samples, result_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default="math")
    parser.add_argument("--prompt_type", type=str, default="tool-integrated")
    parser.add_argument("--file_path", type=str, default=None, required=True)
    parser.add_argument("--max_num_samples", type=int, default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    evaluate(data_name=args.data_name, file_path=args.file_path,
             max_num_samples=args.max_num_samples)
