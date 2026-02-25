# NEXT STEPS — Support Chat Quality Analyzer

This document lists concrete improvements to make the project production-ready,
ordered from highest to lowest priority.

---

## Phase 1 — Core Quality (Do First)

### 1. Add a `validate.py` script
Create `validate.py` that reads `data/chats.json` and checks:
- Each dialog has at least 6 messages
- Roles alternate (client → agent → client …)
- No dialog starts with an agent message
- `lang` field matches the actual language of the text (use `langdetect`)
- `hidden_dissatisfaction: true` dialogs actually end with a client "thank you" signal

**Why:** Ensures the dataset quality before wasting API calls on analysis.

```bash
python validate.py --input data/chats.json
```

---

### 2. Add a `benchmark.py` script
Compare results from two different providers/models on the same dataset:

```bash
python benchmark.py \
  --a results/analysis_deepseek.json \
  --b results/analysis_groq.json \
  --out results/benchmark.md
```

Metrics to compare:
- Agreement rate on `intent` labels
- Agreement rate on `satisfaction`
- Average `quality_score` delta
- Cohen's Kappa for `satisfaction` classification
- Hidden dissatisfaction detection accuracy per model

**Why:** Lets you pick the best model/provider for production use.

---

### 3. Improve hidden dissatisfaction prompting
The current prompt mentions hidden dissatisfaction but does not give the LLM
a dedicated reasoning step. Add a **chain-of-thought** step before the JSON:

```
Before outputting JSON, write 2 sentences in a <think> block:
  - What emotional signals (positive or negative) did you notice in the client's last 2 messages?
  - Does the client's tone match their words?
Then output the JSON.
```

Strip the `<think>` block before JSON parsing.

**Why:** Models that reason first are significantly more accurate at detecting subtle dissatisfaction.

---

### 4. Expand the scenario matrix
Add 3 more scenario topics to `generate.py`:
- `shipping_delay` — order didn't arrive, client escalates
- `data_privacy` — client requests GDPR data deletion
- `onboarding` — new user can't understand how to set up the product

Update `SCENARIOS` list and `SCENARIO_DESC` dict in `generate.py`.

---

## Phase 2 — Better Infrastructure

### 5. Add async/concurrent generation with `asyncio`
Replace the sequential `for` loop in `run_generation()` with concurrent API calls
using `asyncio` + `openai.AsyncOpenAI`. This can cut generation time by 5-10x.

```python
import asyncio
from openai import AsyncOpenAI

async def generate_all(dialogs):
    tasks = [generate_dialog_async(...) for d in dialogs]
    return await asyncio.gather(*tasks)
```

**Note:** Respect provider rate-limits — add a semaphore (e.g. `asyncio.Semaphore(5)`).

---

### 6. Add a proper logging system
Replace all `print()` calls with Python's `logging` module:

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
```

Add `--verbose` / `--quiet` CLI flags.

---

### 7. Add `.env` validation on startup
At the top of both scripts, validate that required dependencies are importable
and that the API key is non-empty before making any calls:

```python
from providers import validate_env
validate_env(provider_name)  # raises SystemExit with a helpful message
```

---

### 8. Add `pyproject.toml` (project metadata + tool config)
Create `pyproject.toml` with:
- `[project]` metadata (name, version, description, authors)
- `[tool.ruff]` — linting rules
- `[tool.mypy]` — type-checking config
- `[tool.pytest.ini_options]` — test discovery config

```toml
[project]
name = "support-chat-analyzer"
version = "0.2.0"
requires-python = ">=3.11"
dependencies = ["openai>=1.30.0", "python-dotenv>=1.0.0"]
```

---

## Phase 3 — Testing

### 9. Add `tests/` directory with pytest
Create the following tests:

| File | What it tests |
|------|---------------|
| `tests/test_providers.py` | `auto_detect_provider()`, `validate_and_clean()` |
| `tests/test_generate.py` | `validate_messages()`, `build_prompt()` |
| `tests/test_analyze.py` | `validate_and_clean()`, `compute_summary()` |
| `tests/test_report.py` | `build_report()` with fixture data |

Use `unittest.mock.patch` to mock API calls — tests must not require real API keys.

```bash
pip install pytest pytest-mock
pytest tests/ -v
```

---

### 10. Add CI with GitHub Actions
Create `.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -r requirements.txt pytest pytest-mock
      - run: pytest tests/ -v
      - run: python -m ruff check .   # linting
```

---

## Phase 4 — UI & Visualization

### 11. Add an HTML report option to `report.py`
Alongside the Markdown report, generate an `results/report.html` with:
- Charts using embedded Chart.js (CDN, no install needed)
- Satisfaction pie chart
- Quality score bar chart
- Scenario breakdown table
- Coloured rows in the full results table (red = score ≤ 2, green = score = 5)

```bash
python report.py --format html --out results/report.html
```

---

### 12. Add a Streamlit dashboard (optional)
Create `dashboard.py` for interactive exploration:

```bash
pip install streamlit
streamlit run dashboard.py
```

Features:
- Load any `analysis.json` file via file picker
- Filter by scenario, satisfaction, case_type
- Expandable dialog viewer (click a row → see full transcript)
- Download filtered results as CSV

---

## Phase 5 — Production Readiness

### 13. Add an `api/` FastAPI server
Wrap the analyzer in a REST endpoint:

```
POST /analyze
Body: { "messages": [ {"role": "client", "text": "..."}, ... ] }
→ { "intent": "...", "satisfaction": "...", "quality_score": 4, ... }
```

Useful for real-time quality monitoring in a production support system.

```bash
pip install fastapi uvicorn
uvicorn api.main:app --reload
```

---

### 14. Add caching layer
Both `generate.py` and `analyze.py` hit the LLM for every dialog.
Add an optional SQLite-backed cache so repeated runs with the same seed/model
return instantly without API calls:

```bash
python analyze.py --cache results/cache.db
```

---

### 15. Write a proper `CONTRIBUTING.md`
Document:
- How to add a new provider (5-line change in `providers.py`)
- How to add a new scenario
- Code style (ruff, mypy)
- PR workflow

---

## Quick Reference — Current CLI

```bash
# 1. Generate dataset (25 dialogs, English)
python generate.py --provider deepseek

# 2. Generate Ukrainian dataset
python generate.py --provider groq --lang uk --count 25

# 3. Resume interrupted generation
python generate.py --provider deepseek --resume

# 4. Analyze all dialogs
python analyze.py --provider deepseek

# 5. Analyze only hidden dissatisfaction dialogs
python analyze.py --provider groq --filter hidden_only

# 6. Analyze only refund dialogs
python analyze.py --filter refund

# 7. Generate Markdown report
python report.py

# 8. Full pipeline with Docker Compose
docker compose run generate
docker compose run analyze
docker compose run report
```
