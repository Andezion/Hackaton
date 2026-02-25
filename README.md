# Support Chat Quality Analyzer

Automated analysis of customer-support chat quality using an LLM.

## Overview

| Script | Purpose |
|--------|---------|
| `generate.py` | Generate a deterministic dataset of realistic support chats |
| `analyze.py` | Analyze each dialog and produce structured quality metrics || `providers.py` | Provider abstraction – DeepSeek, Qwen, OpenAI |

### Supported LLM providers

| Provider | Free tier | Default model | Env variable |
|----------|-----------|---------------|--------------|
| **DeepSeek** ★ | ✅ Yes | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| **Qwen (Alibaba)** | ✅ Yes | `qwen-turbo` | `DASHSCOPE_API_KEY` |
| **Groq** | ✅ Yes | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| **Google Gemini** | ✅ Yes (1500 req/day) | `gemini-2.0-flash` | `GEMINI_API_KEY` |
| **Mistral AI** | ✅ Yes | `mistral-small-latest` | `MISTRAL_API_KEY` |
| **Together AI** | ✅ $25 credits | `Llama-3.3-70B-Instruct-Turbo-Free` | `TOGETHER_API_KEY` |
| **OpenRouter** | ✅ Free models | `llama-3.1-8b-instruct:free` | `OPENROUTER_API_KEY` |
| OpenAI | ❌ Paid | `gpt-4o-mini` | `OPENAI_API_KEY` |

The provider is **auto-detected** from whichever key is present in the environment  
(priority: `deepseek` → `qwen` → `groq` → `gemini` → `mistral` → `together` → `openrouter` → `openai`). You can override it with `--provider`.
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

## Requirements

- Python 3.11+
- A free API key from **one** of the supported providers:
  - DeepSeek: <https://platform.deepseek.com>
  - Qwen/DashScope: <https://dashscope.aliyuncs.com>
  - Groq: <https://console.groq.com>
  - Google Gemini: <https://aistudio.google.com/apikey>
  - Mistral AI: <https://console.mistral.ai>
  - Together AI: <https://api.together.ai>
  - OpenRouter: <https://openrouter.ai>
  - OpenAI (paid): <https://platform.openai.com>

---

## Quick start (local)

```bash
# 1. Clone the repository
git clone <repo-url>
cd support-chat-analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key (choose one provider)
cp .env.example .env
# DeepSeek (recommended – free):
# edit .env → set DEEPSEEK_API_KEY=sk-...
#
# Groq (free, ultra-fast):
# edit .env → set GROQ_API_KEY=gsk_...
#
# Google Gemini (free, 1500 req/day):
# edit .env → set GEMINI_API_KEY=AIza...
#
# Mistral AI (free):
# edit .env → set MISTRAL_API_KEY=...
#
# Qwen/DashScope (free):
# edit .env → set DASHSCOPE_API_KEY=sk-...

# 5. Generate the dataset  (25 dialogs by default)
python generate.py

# 6. Analyze the dataset
python analyze.py
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

```bash
# Use DeepSeek (free, recommended)
python generate.py --provider deepseek

# Use Groq (free, ultra-fast)
python generate.py --provider groq

# Use Google Gemini (free)
python generate.py --provider gemini

# Use Mistral AI (free)
python generate.py --provider mistral

# Use Qwen (free)
python generate.py --provider qwen

# Use a specific model
python generate.py --provider deepseek --model deepseek-reasoner --count 50
```

### analyze.py

| Flag | Default | Description |
|------|---------|-------------|
| `--input PATH` | `data/chats.json` | Input dataset |
| `--out PATH` | `results/analysis.json` | Output file |
| `--provider NAME` | auto-detect | `deepseek` \| `qwen` \| `groq` \| `gemini` \| `mistral` \| `together` \| `openrouter` \| `openai` |
| `--model NAME` | provider default | Override model |
| `--seed N` | `42` | Base seed (determinism) |

```bash
python analyze.py --provider deepseek
python analyze.py --provider groq
python analyze.py --provider gemini --model gemini-1.5-flash
python analyze.py --provider qwen --model qwen-plus
```

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

Or with `docker compose` (set `OPENAI_API_KEY` in your shell or `.env`):

```bash
docker compose run generate
docker compose run analyze
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

## Determinism

Both scripts accept a `--seed` argument that is passed to the OpenAI API
(`seed` parameter) and sets `temperature=0` for analysis. This ensures
reproducible outputs for the same model version.
