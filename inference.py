import os
import pandas as pd
from openai import OpenAI
from env.environment import DataCleaningEnv

# ------------------ ENV VARIABLES ------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ------------------ OPENAI CLIENT ------------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

MAX_STEPS = 6

# ------------------ LOGGING ------------------
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}")

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}")

# ------------------ LLM DECISION ------------------
def get_action_from_llm(dataset,history):
    prompt = f"""
You are an intelligent data cleaning agent.

Your goal is to clean the dataset completely.

You can use the following actions:
- fill_nulls
- remove_nulls
- deduplicate
- convert_types
- trim_whitespace
- normalize

Previous actions:
{history}

Rules:
1. You can choose ANY action.
2. You can choose ANY column.
3. You can repeat actions if needed.
4. You should decide based on dataset issues.
5. Your goal is to maximize data quality.
6. Stop only when dataset is clean.
7. Prefer fixing critical issues first (nulls, duplicates, types)
8. Avoid repeating same action unnecessarily
9.Base your decision ONLY on dataset statistics.
10.Choose different actions depending on issues.

Dataset:
{dataset}

Return ONLY ONE action in this format:
action_type,column_name

Examples:
fill_nulls,city
deduplicate,customer_id
convert_types,age
normalize,income

Do NOT explain anything.
Only return the action.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=50
    )

    output = response.choices[0].message.content.strip()

    try:
        action_type, column = output.split(",")
        return {"type": action_type.strip(), "column": column.strip()}
    except:
        return {"type": "fill_nulls", "column": "city"}  # fallback

# ------------------ MAIN ------------------
def main():
    env = DataCleaningEnv(task=1)
    obs = env.reset()

    rewards = []
    steps_taken = 0
    history = []

    log_start("task1", "data_cleaning", MODEL_NAME)

    for step in range(1, MAX_STEPS + 1):

        df = pd.DataFrame(obs["dataset"])
        col_info = {}

        for col in df.columns:
            col_info[col] = {
                "nulls": float(df[col].isnull().mean()),
                "dtype": str(df[col].dtype),
                "unique": int(df[col].nunique())
            }

        summary = f"""
        Columns: {list(df.columns)}

        Column Info:
        {col_info}

        Duplicates: {df.duplicated().sum()}

        Sample Data:
        {df.head(3).to_dict()}
        """
        action = get_action_from_llm(summary, history)

        # check BEFORE adding
        if str(action) in history:
            action = {"type": "deduplicate", "column": "customer_id"}

        history.append(str(action))

        obs, reward, done, _ = env.step(action)

        rewards.append(reward)
        steps_taken = step

        log_step(step, str(action), reward, done, None)

        if done:
            break

    final = env.submit_cleaned_data(env.dirty_df)
    score = final["final_score"]

    success = score > 0.3

    log_end(success, steps_taken, score, rewards)

if __name__ == "__main__":
    main()