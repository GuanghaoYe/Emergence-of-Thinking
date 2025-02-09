from fastapi import FastAPI, Request
from pydantic import BaseModel

import sys
import io
import os
import json
import signal
from tqdm import tqdm

import inspect
import concurrent
import multiprocessing



def code_extraction(input_text):
    lines = input_text.splitlines()
    code_lines = []
    in_code_block = False

    for line in lines:
        if line.strip() == "```python":  # Start of Python code block
            in_code_block = True
        elif line.strip() == "```":  # End of code block
            if in_code_block:
                break  # End the extraction when the code block ends
        elif in_code_block:
            code_lines.append(line)

    return "\n".join(code_lines)

def read_from_decoded_file(file_path: str):
    examples = []
    with open(file_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            examples.append(example)
    
    return examples

# def evaluate_program(program, test_input, test_output):
#     old_stdout = sys.stdout
#     sys.stdout = io.StringIO()
#     sys.stdin = io.StringIO(test_input)

#     try:
#         def handler():
#             raise TimeoutError("Execution timed out")

#         signal.signal(signal.SIGALRM, lambda signum, frame: handler())
#         signal.alarm(5)  # Set the timeout to 5 seconds

#         try:
#             exec(program, globals())
#             output = sys.stdout.getvalue().strip()
#             if test_output.strip() == output:
#                 return True
#         except TimeoutError as e:
#             output = f"Error: {str(e)}"
#         except Exception as e:
#             output = f"Error: {str(e)}"
#         finally:
#             signal.alarm(0)  # Disable the alarm
#             sys.stdout = old_stdout
#     except Exception as e:
#         output = f"Error: {str(e)}"
#     finally:
#         sys.stdout = old_stdout

#     return False

import concurrent.futures

import multiprocessing
from contextlib import contextmanager

import logging

logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')

# Add debug logs
logging.debug("Starting program execution")


@contextmanager
def redirect_io(input_str: str):
    """上下文管理器，用于重定向标准输入输出"""
    old_stdout = sys.stdout
    old_stdin = sys.stdin
    stdout = io.StringIO()
    stdin = io.StringIO(input_str)
    sys.stdout = stdout
    sys.stdin = stdin
    try:
        yield stdout
    finally:
        sys.stdout = old_stdout
        sys.stdin = old_stdin
        stdout.close()
        stdin.close()

def run_program(conn: multiprocessing.Pipe, 
                program: str, 
                test_input: str) -> None:
    """在独立进程中运行程序"""
    try:
        with redirect_io(test_input) as stdout:
            # 创建一个新的全局命名空间来执行代码
            local_globals = {}
            # 执行程序
            exec(program, local_globals)
            # 获取输出
            output = stdout.getvalue().strip()
            conn.send(output)
    except Exception as e:
        logging.error(f"Program execution error: {str(e)}")
        conn.send(f"Error: {str(e)}")
    finally:
        conn.close()

def evaluate_program(program: str, 
                    test_input: str, 
                    test_output: str, 
                    timeout: int = 5) -> bool:
    """
    评估程序是否正确执行并产生预期输出
    
    Args:
        program: 要执行的Python代码字符串
        test_input: 测试输入
        test_output: 期望的输出
        timeout: 执行超时时间（秒）
    
    Returns:
        bool: 程序输出是否匹配预期输出
    """
    logging.debug(f"Starting evaluation with input: {test_input}")
    
    # 创建管道用于进程间通信
    parent_conn, child_conn = multiprocessing.Pipe()
    
    # 创建进程
    process = multiprocessing.Process(
        target=run_program,
        args=(child_conn, program, test_input)
    )
    
    try:
        # 启动进程
        process.start()
        logging.debug(f"Started process with PID: {process.pid}")
        
        # 等待结果，设置超时
        if parent_conn.poll(timeout):
            # 接收输出
            output = parent_conn.recv()
            logging.debug(f"Received output: {output}")
            
            # 检查是否是错误消息
            if isinstance(output, str) and output.startswith("Error:"):
                logging.error(f"Program execution failed: {output}")
                return False
                
            # 比较输出
            result = test_output.strip() == output.strip()
            logging.debug(f"Output comparison result: {result}")
            return result
        else:
            logging.warning("Program execution timed out")
            return False
            
    except Exception as e:
        logging.error(f"Evaluation error: {str(e)}")
        return False
        
    finally:
        # 清理资源
        logging.debug("Cleaning up resources...")
        parent_conn.close()
        child_conn.close()
        
        # 确保进程被终止
        if process.is_alive():
            logging.debug("Terminating process...")
            process.terminate()
            process.join(timeout=1)
            
            # 如果进程仍然活着，强制结束它
            if process.is_alive():
                logging.warning("Force killing process...")
                if sys.platform != 'win32':  # Windows不支持SIGKILL
                    os.kill(process.pid, signal.SIGKILL)
                else:
                    os.kill(process.pid, signal.CTRL_C_EVENT)
                    
            process.join()
        
        logging.debug("Evaluation completed")



app = FastAPI()
    

def parse_query(sequence):
    # Parse the query
    global args
    if 'qwen' in args.model_name.lower():
        query = sequence.split("\n\nplease only reply with the source code in python.")[0].split('<|im_start|>user\n')[1].strip()
        response = sequence.split("<|im_start|>assistant\n")[1].strip()
    elif 'phi' in args.model_name.lower():
        query = sequence.split("\n\nplease only reply with the source code in python.")[0].strip('<|user|>').strip()
        response = sequence.split("<|assistant|>")[1].strip()
    elif 'llama' in args.model_name.lower():
        query = sequence.split("\n\nplease only reply with the source code in python.")[0].split('user<|end_header_id|>')[1].strip()
        response = sequence.split("assistant<|end_header_id|>\n")[1].strip()
    else:
        raise ValueError("Model name not recognized")
    return query, response

def calculate_reward(data: dict) -> float | list[float]:
    # Simple reward calculation based on the length of query and response
    rewards = []
    sequence = data.get("query", "")
    if isinstance(sequence, str):
        sequence = [sequence]
    for s in sequence:
        try:
            # query = s.split("\n\nplease only reply with the source code in python.")[0].strip('<|user|>').strip()
            # response = s.split("<|assistant|>")[1].strip()

            query, response = parse_query(s)

            test_cacaes = problem_test_cases_dict[query]
            program = code_extraction(response)
            correct = 0
            total = 0
            for test_input, test_output in zip(test_cacaes['input'], test_cacaes['output']):
                if test_input.strip() in query:
                    # print("Skipping test case as input is in query")
                    continue
                if evaluate_program(program, test_input, test_output):
                    correct += 1
            
                total += 1
            reward = correct / total
        except Exception as e:
            # print(f"Error2: {str(e)}")
            reward = 0.0
        rewards.append(reward)
    return rewards

@app.post("/get_reward")
async def get_reward(data: Request):
    try:
        data = await data.json()
        reward = calculate_reward(data)
    except Exception as e:
        print(f"Error1: {str(e)}")
        reward = 0.0
    return {"rewards": reward}

if __name__ == '__main__':
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser(description='Run the Code Contests Reward Model server')
    parser.add_argument('--context_file', type=str, help='Path to the decoded file', required=True)
    parser.add_argument('--model_name', type=str, help='model name', required=True)

    args = parser.parse_args()

    problem_test_cases = read_from_decoded_file(args.context_file)

    problem_test_cases_dict = {}

    for example in tqdm(problem_test_cases, desc="Running tests on decoded data", total=len(problem_test_cases)):
        query = example['description']
        
        public_test = example['public_tests']
        generated_test = example['generated_tests']
        all_test_input = []
        all_test_output = []
        for test_input, test_output in zip(public_test['input'], public_test['output']):
            all_test_input.append(test_input)
            all_test_output.append(test_output)
        for test_input, test_output in list(zip(generated_test['input'], generated_test['output']))[:3]:
            all_test_input.append(test_input)
            all_test_output.append(test_output)
        problem_test_cases_dict[query] = {'input': all_test_input, 'output': all_test_output}

    uvicorn.run(app, host="0.0.0.0", port=8000)