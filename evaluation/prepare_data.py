"""
Evaluate an LLM on a dataset. Use data parallelism (multiple model replica)
Therefore, if 8 gpus are used, then 8 models are created and each model processes 1/8 of the dataset.
Does not support tensor paralelism due to a bug.
TODO: support both data paralellism and model parallelism.
"""

import os
import random
import argparse
import time
import json
import logging
import multiprocessing
import traceback
import pathlib
import functools

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import torch
from vllm.lora.request import LoRARequest

from omegaconf import OmegaConf

from evaluation import data_loader
from evaluation import parser
from evaluation import model_utils
from evaluation import python_executor
from evaluation import trajectory
from evaluation import utils
from evaluation import evaluate

logger = logging.getLogger("eval_math_data_parallel.py")
logger.setLevel(logging.INFO)
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    parser.add_argument("--async_evaluate", default=False, action="store_true")
    parser.add_argument("--message_path", default=None, type=str)
    parser.add_argument("--config", default=None, type=str)

    args = parser.parse_args()

    if args.config is not None:
        # Load the YAML configuration file
        yaml_config = OmegaConf.load(args.config)

        # Merge argparse arguments with the YAML configuration file.
        # NOTE: config file takes precedence, there other arguments provided in command line will be ignored.
        args = OmegaConf.merge(OmegaConf.create(vars(args)), yaml_config)

    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)

    # get the number of available GPUs
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        args.data_parallel_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    else:
        args.data_parallel_size = torch.cuda.device_count()

    return args

def is_choice(answer):
    """
    whether answer is multiple-choice
    """
    return answer in ["A", "B", "C", "D", "E"]

def is_multi_choice(answer):
    """
    whether answer is multiple-choice and check-all-that-apply
    """
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def prepare_data(data_name, args, parallel_index=None):
    """
    1. load data
    # 2. sample `num_test_sample` from dataset
    3. shuffle
    4. select start and end
    5. get out_file name
    6. if there are processed samples, load them and deduplicate
    """
    examples = data_loader.load_data(data_name, args.split, args.data_dir)

    examples = [examples[args.num_test_sample]]

    # shuffle
    if args.shuffle:
        # random.seed(datetime.now().timestamp())
        random.seed(args.seed) # fix the seed across different processes
        random.shuffle(examples)

    # select start and end
    if parallel_index is not None:
        examples = examples[
            parallel_index * len(examples) // args.data_parallel_size : (parallel_index + 1)
            * len(examples)
            // args.data_parallel_size
        ]
    else:
        examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    return examples

# def async_evaluate(message_path):
#     checkpoints = []
#     while True:
#         content = json.load(open(message_path, 'r'))['evaluator']
#         if content['status'] == 'finished':
#             break
#         remaining_checkpoints = [checkpoint for checkpoint in content['checkpoints'] if checkpoint not in checkpoints]
#         if len(remaining_checkpoints) > 0:
#             remaining_checkpoints = sorted(remaining_checkpoints, key=lambda x: x['time_stamp'])
#             for checkpoint in remaining_checkpoints:
#                 # subprocess.run(['python', '-u', '-m', 'evaluation.eval_math_data_parallel', '--config', config_path])
#                 args.model_name_or_path = checkpoint['path']
#                 setup(args)
#                 checkpoints.append(checkpoint)

def setup(args):
    """
    set up data parallel gpus   
    iterate dataset in data_list
    """
    # load model

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:

        # NOTE: get out_file name
        # dt_string = datetime.now().strftime("%m-%d_%H-%M")
        model_name = "/".join(args.model_name_or_path.split("/")[-2:]) # take last 2 parts of the path
        out_file_prefix = f"{args.split}_{args.prompt_type}_seed{args.seed}_t{args.temperature}"
        output_dir = args.output_dir
        # if not os.path.exists(output_dir):
        #     output_dir = f"outputs/{output_dir}"
        os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

        llm, tokenizer = prepare_model(args)
        examples = data_loader.load_data(data_name, args.split, args.data_dir)

        for i in range(2730):
            if i < 74:
                continue

            out_file = f"{output_dir}/{data_name}/{model_name}/{i}.jsonl"
            
            args.out_file = out_file

            lll = examples[i*100:(i+1)*100]

            # NOTE generate
            if args.data_parallel_size == 1:
                all_samples, time_use = generate(llm, tokenizer, lll, data_name, args, 0)
                time_use = [time_use]

            else:
                try:
                    pool = multiprocessing.Pool(processes=args.data_parallel_size)
                    generate_results = pool.starmap(generate, [(data_name, args, parallel_index) for parallel_index in range(args.data_parallel_size)])

                    # Concatenate the results
                    all_samples = [item for split in generate_results for item in split[0]]
                    time_use = [split[1] for split in generate_results]

                except Exception as e:
                    print(f"An error occurred: {e}")
                    traceback.print_exc()
                finally:
                    pool.close()
                    pool.join()


            # sort and deduplicate
            all_samples = sorted(all_samples, key=lambda x: x["idx"])
            all_samples = [all_samples[i] for i in range(len(all_samples)) if i == 0 or all_samples[i]["idx"] != all_samples[i - 1]["idx"]]

            logger.info(len(all_samples))

            # evaluate
            all_samples, result_json = evaluate.evaluate(
                samples=all_samples,
                data_name=data_name,
            )

            for sss in all_samples:
                del sss["code"], sss["pred"], sss["report"], sss["gt_cot"], sss["gt"]

            utils.save_jsonl(all_samples, out_file)

            print(f"Finished {i*100} samples")

            # average_time_use = np.mean(time_use)
            # result_json["average_time_use_in_second"] = average_time_use
            # result_json["averate_time_use_in_minite"] = (
            #     f"{int(average_time_use // 60)}:{int(average_time_use % 60):02d}"
            # )

            # with open(
            #     out_file.replace(".jsonl", f"_metrics.json"), "w"
            # ) as f:
            #     json.dump(result_json, f, indent=4)

            # results.append(result_json)

    # add "avg" result to data_list and results
    # data_list.append("avg")
    # results.append(
    #     {
    #         "maj_acc": sum([result["maj_acc"] for result in results]) / len(results),
    #     }
    # )

    # # print all results
    # pad = max([len(data_name) for data_name in data_list])
    # print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    # print("\t".join([f"{result['maj_acc']:.3f}".ljust(pad, " ") for result in results]))

def prepare_model(args):
    """
    load model and tokenizer either vllm or huggingface
    """
    if args.use_vllm:
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
        llm = model_utils.load_vllm_with_lora(
            model_name_or_path=args.model_name_or_path,
            gpus=[0],
            seed=args.seed,
            # dtype='bfloat16',
        )

    else:
        llm, tokenizer = model_utils.load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            # use_safetensors=args.use_safetensors,
            use_safetensors=True,
        )
    
    return llm, tokenizer

def generate(llm, tokenizer, examples, data_name, args, parallel_index=None):
    """
    1. prepare data
    2. prepare executor
    4. parse question and answer, add fields, repeat prompts, apply template, stop words
    5. generate (multiple epochs if tora type prompt)
    6. if needed, execute code, sort into remain_prompts and end_prompts.
    """
    # set the gpu index to use for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(parallel_index)

    # print("=" * 50)
    # print("data:", data_name, " ,remain samples:", len(examples))
    # if len(examples) > 0:
    #     print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = python_executor.PythonExecutor(get_answer_expr="solution()")
    else:
        executor = python_executor.PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parser.parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parser.parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = parser.construct_prompt(example, data_name, args)

        # if idx == args.start:
        #     print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [
        sample["prompt"] for sample in samples for _ in range(args.n_sampling)
    ]
    # apply template
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    remain_prompts = input_prompts
    remain_prompts = [(i, prompt) for i, prompt in enumerate(remain_prompts)]
    end_prompts = []

    # support tora type prompt.
    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    # add stop words
    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        # print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        # get all outputs
        prompts = [item[1] for item in current_prompts]
        if args.use_vllm:
            outputs = llm( # LLM.generate
                prompts,
                SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens_per_call,
                    n=1,
                    stop=stop_words,
                    stop_token_ids=(
                        [151645, 151643]
                        if "qwen2" in args.model_name_or_path.lower()
                        else None
                    ),
                ),
            )

            outputs = sorted(
                outputs, key=lambda x: int(x.request_id)
            )  # sort outputs by request_id
            outputs = [output.outputs[0].text for output in outputs]
        else:
            outputs = model_utils.generate_completions(
                model=llm,
                tokenizer=tokenizer,
                prompts=prompts,
                max_new_tokens=args.max_tokens_per_call,
                batch_size=16,
                stop_id_sequences=stop_words,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=(args.temperature > 0),
            )

        assert len(outputs) == len(current_prompts)

        # process all outputs
        remain_prompts = []
        remain_codes = []
        for (i, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, query))
                if "```python" in output:
                    output = trajectory.extract_program(query)
                remain_codes.append(output)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            elif "boxed" not in output and output.endswith("```"):
                program = trajectory.extract_program(query)
                remain_prompts.append((i, query))
                remain_codes.append(program)
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, query)

    # unsolved samples
    # print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        parser.run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling : (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        for j in range(len(preds)):
            if is_choice(sample["gt"]) and not is_choice(preds[j]):
                preds[j] = parser.choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        sample.update({"code": code, "pred": preds, "report": reports})

        all_samples.append(sample)
    
    return all_samples, time_use

if __name__ == "__main__":
    args = parse_args()
    utils.set_seed(args.seed)
    setup(args)
    # if args.async_evaluate:
    #     async_evaluate(args)
    # else:
    #     setup(args)
