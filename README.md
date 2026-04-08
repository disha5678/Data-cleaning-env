# 🧠 AI-Powered Data Cleaning Environment (OpenEnv)

## 📌 Overview

This project implements an **AI-driven data cleaning environment** following the OpenEnv specification.  
An LLM-based agent interacts with the environment step-by-step and performs data cleaning actions to improve dataset quality.

The system evaluates:
- Decision-making ability of AI
- Data cleaning strategies
- Efficiency (steps vs quality)

---

## 🎯 Objective

To build an environment where an AI agent can:
- Detect data issues
- Apply cleaning actions
- Maximize final data quality score

---

## ⚙️ Environment Design

### 🔹 Observation Space

Each step returns:

```json
{
  "dataset": {...},
  "shape": [rows, columns],
  "steps": n
}
```

- `dataset`: current dataset state
- `shape`: dimensions
- `steps`: steps taken so far

### 🔹 Action Space

Agent can perform:

| Action | Description |
|--------|-------------|
| `fill_nulls` | Fill missing values |
| `remove_nulls` | Remove rows with nulls |
| `deduplicate` | Remove duplicate rows |
| `convert_types` | Fix incorrect data types |
| `trim_whitespace` | Clean text formatting |
| `normalize` | Normalize numeric columns |
| `inspect_column` | Analyze a column |

**Action Format:**

```json
{
  "type": "action_name",
  "column": "column_name"
}
```

### 🔹 Reward System

- Positive reward → correct cleaning
- Negative reward → unnecessary/wrong action

**Example:**
- Fill nulls → `+0.12`
- Wrong removal → `-0.08`

### 🔹 Episode Termination

Episode ends when:
- `done = True` OR
- max steps reached

Final score is computed using:

```
score ∈ [0, 1]
```

---

## 🧪 Tasks

### ✅ Task 1: Basic Cleaning
- Handle null values
- Remove duplicates
- Fix data types

### ✅ Task 2: Intermediate Cleaning
- Better decision strategies
- Column-wise reasoning

### ✅ Task 3: Full Pipeline
- Complete dataset cleaning
- Optimal sequence of actions

---

## 🤖 AI Agent (Inference)

The agent uses an LLM to:
1. Analyze dataset summary
2. Choose next action
3. Avoid repeating actions
4. Improve data quality iteratively

### 🔹 Strategy Used

Instead of sending full dataset, we send:
- Column statistics (null %, dtype, unique values)
- Duplicate count
- Sample rows
- Action history

👉 This improves reasoning and reduces noise.

---

## 📊 Baseline Performance

| Task | Score |
|------|-------|
| Task 1 | ~0.70 |
| Task 2 | ~0.75 |
| Task 3 | ~0.80 |

---

## 🚀 Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <repo-url>
cd data-cleaning-env
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set Environment Variables

**Windows (PowerShell):**

```powershell
setx HF_TOKEN "your_token_here"
setx MODEL_NAME "Qwen/Qwen2.5-72B-Instruct"
setx API_BASE_URL "https://router.huggingface.co/v1"
```

**Linux/Mac:**

```bash
export HF_TOKEN="your_token_here"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export API_BASE_URL="https://router.huggingface.co/v1"
```

### 4️⃣ Run Inference

```bash
python inference.py
```

---

## 📡 API Endpoints

### 🔹 Reset

```
POST /reset
```

### 🔹 Step

```
POST /step
```

**Example:**

```json
{
  "type": "fill_nulls",
  "column": "city"
}
```

### 🔹 State

```
GET /state
```

---

## 🐳 Docker Setup

**Build:**

```bash
docker build -t data-cleaning-env .
```

**Run:**

```bash
docker run -p 7860:7860 data-cleaning-env
```

---

## 🌐 Hugging Face Deployment

1. Create Space → Docker
2. Push code
3. Add environment variables:
   - `HF_TOKEN`
   - `MODEL_NAME`
   - `API_BASE_URL`

---

## ✅ Validation

```bash
openenv validate
```

```bash
bash validate-submission.sh <your-space-url>
```

---

## 📁 Project Structure

```
.
├── app.py
├── inference.py
├── Dockerfile
├── requirements.txt
├── README.md
└── env/
    ├── environment.py
    ├── actions.py
    ├── data_generator.py
    ├── issue_injector.py
    └── graders/
```

---

## ⚠️ Constraints

- Runtime < 20 minutes
- Compatible with:
  - 2 vCPU
  - 8GB RAM
- Must follow OpenEnv spec

---

## 🎉 Conclusion

This project demonstrates:
- AI-based decision making
- Reinforcement-style environment
- Automated data cleaning

---

## 👩‍💻 Authors

- Hackathon Team