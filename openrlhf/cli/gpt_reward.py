import json
import os
import random
import re
import time

import openai


def _get_openai_client():
    openai_keys = [value for key, value in os.environ.items() if key.lower().startswith("openai_api_key")]

    if random.random() < 0.05:
        key = random.choice(openai_keys) if openai_keys else None
    else:
        key = os.environ.get("OPENAI_API_KEY")

    return openai.OpenAI(api_key=key)

def _extract_score_from_response(response: str) -> float:
    """
    Extracts a numeric score from a text response. First tries to find a
    triple-backtick JSON block. If that fails, it uses a regex fallback
    that allows optional quotes around 'score' or 'grade'.

    Args:
      response (str): The text that might contain JSON or a score/grade.

    Returns:
      float: The extracted score or -1 if none is found.
    """
    score = -1.0
    try:
        # Look for JSON between ```json\n and \n```
        grading_output = re.search(
            r'(?<=```json\n).*?(?=\n```)',
            response,
            re.DOTALL
        ).group(0)
        grading_data = json.loads(grading_output)
        score = grading_data.get("grade", None)
        if score is None:
            score = grading_data.get("score", None)
        if score is None:
            score = -1.0
    except Exception:
        # Use a regex that allows optional quotes around
        # 'score' or 'grade', and optional quotes around the number.
        match = re.search(
            r'(?i)[\'"]?(score|grade)[\'"]?\s*:\s*[\'"]?(\d+(\.\d+)?)[\'"]?',
            response
        )
        if match:
            score = float(match.group(2))
    return float(score)

def _get_client(api_type: str, endpoint_url: str = None):
    if api_type == "azure":
        if not endpoint_url:
            raise ValueError("For Azure API, endpoint_url must be provided.")
        client = openai.AzureOpenAI(
            azure_endpoint=endpoint_url,
            api_key=os.environ["AZURE_OPENAI_KEY"],
            api_version="2024-02-01",
        )
    elif api_type == "openai":
        client = _get_openai_client()
    else:
        raise ValueError("Invalid API type. Must be 'openai' or 'azure'.")
    return client

def grade_cot(problem: str, answer: str, grader_model_name: str = "gpt-4o-mini", 
              api_type: str = "openai", 
              endpoint_url: str = "") -> float:
    """
    Grades a solution using a specified OpenAI or Azure OpenAI model.

    Args:
        problem (str): The problem statement to evaluate.
        answer (str): The solution provided for grading.
        grader_model_name (str): The name of the model to use for grading. Defaults to "gpt-4o-mini".
        api_type (str): The API type to use, either "openai" or "azure". Defaults to "openai".
        endpoint_url (str, optional): The endpoint URL required for Azure OpenAI API. Defaults to None.

    Returns:
        float: A score between 0 and 1 representing the quality of the solution. Returns -1 if an error occurs.
    """
    max_retries = 10
    backoff_factor  = 2 # seconds

    # Construct messages for the API call
    messages = [
        {"role": "system", "content": THINKING_SYS_MSG},
        {"role": "user", "content": f"Problem: {problem}\n\n---\n\nSolution: {answer}"}
    ]

    for attempt in range(max_retries):
        try:
            client = _get_client(api_type, endpoint_url)

            # Make the API call
            response = client.chat.completions.create(
                messages=messages,
                model=grader_model_name,
            )
            break # Success! exit the loop and grade the response
        except Exception as e:
            print(f"Error during API call (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:  # If retries remain
                sleep_time = backoff_factor * (2 ** attempt)
                print(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                print("Max retries reached. Returning -1.")
                return -1
    try:
        # Extract the score from the response
        grading_output = response.choices[0].message.content
        # print("GPT REWARD RESPONSE:", grading_output)

        score = _extract_score_from_response(grading_output)

        # Ensure the score is within the expected range [0, 1]
        if 0.0 <= score <= 1.0:
            return score
        else:
            return -1

    except Exception as e:
        print(f"Error during grading: {e}")
        return -1


THINKING_SYS_MSG = """
You are a **Thinking-Effort Grading Assistant**. Your goal is to assess a solution’s thinking trajectory and output a single numeric score in the range **[0,1]** based on how hard the solver tried. You must **not** evaluate correctness of the final answer. Instead, you will grade the solution’s approach on aspects such as creativity, thoroughness, exploration of different methods, and evidence of “thinking outside the box.”

Use the following steps and guidelines:

---

### 1. Understand the Inputs

- **Problem Statement**: A description of the task or question the solver was trying to address.  
- **Solution Trajectory**: The step-by-step reasoning, sketches, or approaches the solver used.

You will be given both pieces of information. You do **not** need to verify correctness of the solution; your focus is on the process and the effort.

---

### 2. Key Dimensions to Evaluate

1. **Diversity of Strategies**  
   - How many different approaches or angles did the solver consider?  
   - Did they pivot or switch methods after encountering difficulties?

2. **Depth of Exploration**  
   - Did the solver provide detailed steps or partial progress?  
   - Did they elaborate on the reasoning behind each step, showing a genuine effort to tackle the problem?

3. **Creativity and Novelty**  
   - Did the solver propose any unusual or “out-of-the-box” ideas?  
   - Were there any signs of creative leaps or innovative methods?

4. **Persistence and Rigor**  
   - Did the solver systematically test, refine, or discard ideas?  
   - Did they keep trying to move forward despite challenges or dead ends?

---

### 3. Scoring Rubric

Use the following guidelines to translate the above dimensions into a single numeric score from **0** to **1**:

- **Score = 0.0**  
  - The solver provided almost no indication of effort.  
  - Their solution trajectory is extremely short, with no exploration of strategies.

- **Score = 0.2 – 0.4**  
  - The solver did some minimal exploration or attempts.  
  - They might have tried only one strategy, or provided very little reasoning.

- **Score = 0.5 – 0.7**  
  - The solver showed moderate effort, exploring at least a couple of approaches or providing some detail.  
  - They might have tried to reason through steps but only partially demonstrated creativity or persistence.

- **Score = 0.8 – 0.9**  
  - The solver’s trajectory was fairly thorough, featuring multiple strategies, iteration, and some creativity.  
  - They clearly tried to refine or re-think aspects of their approach.

- **Score = 1.0**  
  - The solver demonstrated extensive exploration with varied methods, significant detail, creativity, and tenacity.  
  - They showed strong persistence and repeatedly revisited or innovated their strategies.

---

### 4. Output Format

Return your final evaluation in **JSON** format, containing:

- **rationale**: A concise explanation (one to three sentences) justifying why you selected that score based on the above criteria.
- **grade**: A floating-point value in the range [0,1].  

**Example**:
```json
{
  "rationale": "The solver explored multiple approaches and provided detailed reasoning steps. However, there was limited evidence of truly out-of-the-box creativity."
  "grade": 0.75,
}
```

---

### 5. Constraints and Notes

- You must **not** critique or judge the **correctness** of the solution.  
- Focus only on the **process**, effort, and creativity observed.  
- Ensure that your numeric score properly reflects the dimensions outlined above.  
- Provide a clear and concise **rationale** that references key observations about the solver’s trajectory.

"""

if __name__ == "__main__":
    ans = grade_cot("What is 2 + 5?", "Primary school math. Adding two apples and two apples. It is four apples. It is 4.", "gpt-4o-mini", "openai")
    print(ans)
    import pdb; pdb.set_trace()


