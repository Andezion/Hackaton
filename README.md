# Support Chat Quality Analyzer

Automated analysis of customer-support chat quality using an LLM.

## Overview

| Script | Purpose |
|--------|---------|
| `generate.py` | Generate a deterministic dataset of realistic support chats |
| `analyze.py` | Analyze each dialog and produce structured quality metrics |
| `providers.py` | Provider abstraction – DeepSeek, Qwen, OpenAI |
|`run.py` | Runs all code and shows results |

### What is analyzed

| Field | Values |
|-------|--------|
| `intent` | `payment_issue` · `technical_error` · `account_access` · `tariff_question` · `refund` · `other` |
| `satisfaction` | `satisfied` · `neutral` · `unsatisfied` |
| `quality_score` | `1` (worst) – `5` (best) |
| `agent_mistakes` | `ignored_question` · `incorrect_info` · `rude_tone` · `no_resolution` · `unnecessary_escalation` |

The system detects **hidden dissatisfaction** – cases where the client formally
thanks the agent but the underlying problem remains unresolved.


---

## Quick start (local)

```bash
git clone <repo-url>
cd support-chat-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate          

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key (choose one provider)
cp .env.example .env

# 5. Generate the dataset  (25 dialogs by default)
python generate.py

# 6. Analyze the dataset
python run.py
```

Results are written to:
- `data/chats.json`      – generated dialogs
- `results/analysis.json` – per-dialog metrics + summary

---

## CLI options

### generate.py

| Flag | Default | Description |
|------|---------|-------------|
| `--count N` | `25` | Number of dialogs to generate |
| `--out PATH` | `data/chats.json` | Output file |
| `--provider NAME` | auto-detect | `deepseek` \| `qwen` \| `groq` \| `gemini` \| `mistral` \| `together` \| `openrouter` \| `openai` |
| `--model NAME` | provider default | Override model |
| `--seed N` | `42` | Base seed (determinism) |


### analyze.py

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | `data/chats.json` | Input dataset |
| `--out PATH` | `results/analysis.json` | Output file |
| `--provider NAME` | auto-detect | `deepseek` \| `qwen` \| `groq` \| `gemini` \| `mistral` \| `together` \| `openrouter` \| `openai` |
| `--model NAME` | provider default | Override model |
| `--seed N` | `42` | Base seed (determinism) |



---

## Docker

```bash
# Build the image
docker build -t support-analyzer .

# Generate dataset
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v "$(pwd)/data:/app/data" \
  support-analyzer python generate.py

# Analyze dataset
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/results:/app/results" \
  support-analyzer python analyze.py
```

---

## Output format

### data/chats.json

```json
[
  {
    "id": 1,
    "scenario": "payment_issue",
    "case_type": "successful",
    "hidden_dissatisfaction": false,
    "seed": 42,
    "messages": [
      {"role": "client", "text": "Hi, I was charged twice for my subscription..."},
      {"role": "agent",  "text": "I'm sorry to hear that! Let me look into it..."}
    ]
  }
]
```

### results/analysis.json

```json
{
  "model": "gpt-4o-mini",
  "seed": 42,
  "summary": {
    "total_dialogs": 25,
    "avg_quality_score": 3.24,
    "intent_distribution": {"payment_issue": 5, "technical_error": 5, "...": "..."},
    "satisfaction_distribution": {"satisfied": 10, "neutral": 8, "unsatisfied": 7},
    "mistake_frequency": {"no_resolution": 7, "ignored_question": 4, "...": "..."}
  },
  "results": [
    {
      "dialog_id": 1,
      "scenario": "payment_issue",
      "case_type": "successful",
      "hidden_dissatisfaction": false,
      "intent": "payment_issue",
      "satisfaction": "satisfied",
      "quality_score": 5,
      "agent_mistakes": [],
      "reasoning": "The agent resolved the duplicate charge promptly and apologized sincerely."
    }
  ]
}
```

---

## Dataset design

The scenario matrix covers 5 topics × 5 case types, including:

| Scenario | Case types |
|----------|-----------|
| `payment_issue` | successful, problematic, conflict, agent_error, hidden dissatisfaction |
| `technical_error` | successful, problematic, conflict, agent_error, hidden dissatisfaction |
| `account_access` | successful, problematic, conflict, agent_error, hidden dissatisfaction |
| `tariff_question` | successful, problematic, conflict, agent_error, hidden dissatisfaction |
| `refund` | successful, problematic, conflict, agent_error, hidden dissatisfaction |

**Hidden dissatisfaction** dialogs end with the client politely thanking the
agent while the core issue is still unresolved – designed to test the
analyzer's ability to detect subtle signals.

---

