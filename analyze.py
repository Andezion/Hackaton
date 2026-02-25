"""
analyze.py
----------
Analyzes each support-chat dialog in a dataset and produces structured
quality metrics using an LLM.

Output per dialog:
  - dialog_id        : int
  - intent           : payment_issue | technical_error | account_access |
                       tariff_question | refund | other
  - satisfaction     : satisfied | neutral | unsatisfied
  - quality_score    : int 1–5
  - agent_mistakes   : list[ ignored_question | incorrect_info | rude_tone |
                             no_resolution | unnecessary_escalation ]
  - reasoning        : short free-text justification (for transparency)
  - elapsed_ms       : LLM call duration in milliseconds
  - analyzed_at      : ISO-8601 UTC timestamp

Extra summary fields:
  - hidden_dissatisfaction_accuracy: how often the LLM correctly flagged
    hidden dissatisfaction dialogs as "unsatisfied"
  - scenario_breakdown: avg quality_score and satisfaction distribution per scenario

Usage:
    python analyze.py [--input data/chats.json] [--out results/analysis.json]
                      [--provider PROVIDER] [--model MODEL] [--seed 42]
                      [--filter all|hidden_only|<scenario>]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from providers import PROVIDERS, auto_detect_provider, get_client

load_dotenv()

# ─────────────────────────────────────────────
# Allowed enum values (used both in the prompt and for validation)
# ─────────────────────────────────────────────

VALID_INTENTS = {
    "payment_issue",
    "technical_error",
    "account_access",
    "tariff_question",
    "refund",
    "other",
}

VALID_SATISFACTION = {"satisfied", "neutral", "unsatisfied"}

VALID_MISTAKES = {
    "ignored_question",
    "incorrect_info",
    "rude_tone",
    "no_resolution",
    "unnecessary_escalation",
}

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a quality-assurance analyst for a customer support team.
Your task is to evaluate support-chat transcripts and return a structured JSON assessment.
Be objective and precise. Pay special attention to HIDDEN dissatisfaction:
the client may formally thank the agent while the underlying issue remains unresolved –
look for subtle cues (terse replies, vague acceptance, absence of genuine relief).

Always respond with valid JSON only – no markdown fences, no extra keys."""

ANALYSIS_SCHEMA = """{
  "intent": "<one of: payment_issue | technical_error | account_access | tariff_question | refund | other>",
  "satisfaction": "<one of: satisfied | neutral | unsatisfied>",
  "quality_score": <integer 1–5>,
  "agent_mistakes": ["<zero or more of: ignored_question | incorrect_info | rude_tone | no_resolution | unnecessary_escalation>"],
  "reasoning": "<2–4 sentences explaining your rating>"
}"""

QUALITY_RUBRIC = """
Quality score rubric:
  5 – Issue fully resolved, agent is empathetic, accurate, and efficient.
  4 – Issue resolved with minor delays or minor tone issues.
  3 – Partial resolution or mild professionalism issues.
  2 – Issue unresolved or notable errors (wrong info, question ignored, etc.).
  1 – Severe failure: rude tone, multiple errors, escalation mishandled, or client clearly worse off.

Satisfaction rubric:
  satisfied    – client's issue is genuinely resolved AND the client's tone is positive.
  neutral      – issue partially resolved or client is polite but unenthusiastic.
  unsatisfied  – issue unresolved, client is frustrated, OR hidden dissatisfaction detected.
"""

MISTAKE_DEFINITIONS = """
Mistake definitions:
  ignored_question       – agent failed to address a direct question from the client.
  incorrect_info         – agent provided factually wrong information.
  rude_tone              – agent was dismissive, sarcastic, or impolite.
  no_resolution          – conversation ended without solving or clearly escalating the issue.
  unnecessary_escalation – agent escalated (e.g., forwarded to supervisor, asked client to call) 
                           when the issue could have been handled directly.
"""


def build_analysis_prompt(dialog: dict) -> str:
    # Format the transcript
    lines = []
    for msg in dialog["messages"]:
        role = msg["role"].upper()
        lines.append(f"{role}: {msg['text']}")
    transcript = "\n".join(lines)

    return (
        f"Analyze the following support-chat transcript.\n\n"
        f"=== TRANSCRIPT ===\n{transcript}\n=== END ===\n\n"
        f"{QUALITY_RUBRIC}\n{MISTAKE_DEFINITIONS}\n"
        f"Return ONLY a JSON object matching this schema:\n{ANALYSIS_SCHEMA}"
    )


# ─────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────

def validate_and_clean(raw: dict) -> dict:
    """Validate LLM output fields and coerce to known values where possible."""
    intent = str(raw.get("intent", "other")).strip().lower()
    if intent not in VALID_INTENTS:
        intent = "other"

    satisfaction = str(raw.get("satisfaction", "neutral")).strip().lower()
    if satisfaction not in VALID_SATISFACTION:
        satisfaction = "neutral"

    try:
        quality_score = int(raw.get("quality_score", 3))
        quality_score = max(1, min(5, quality_score))
    except (TypeError, ValueError):
        quality_score = 3

    raw_mistakes = raw.get("agent_mistakes", [])
    if not isinstance(raw_mistakes, list):
        raw_mistakes = []
    agent_mistakes = [
        m.strip().lower()
        for m in raw_mistakes
        if isinstance(m, str) and m.strip().lower() in VALID_MISTAKES
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    agent_mistakes = [m for m in agent_mistakes if not (m in seen or seen.add(m))]  # type: ignore[func-returns-value]

    reasoning = str(raw.get("reasoning", "")).strip()

    return {
        "intent": intent,
        "satisfaction": satisfaction,
        "quality_score": quality_score,
        "agent_mistakes": agent_mistakes,
        "reasoning": reasoning,
    }


# ─────────────────────────────────────────────
# LLM call
# ─────────────────────────────────────────────

def analyze_dialog(
    client,
    cfg,
    dialog: dict,
    model: str,
    seed: int,
) -> dict:
    """Analyze a single dialog and return validated metrics."""
    prompt = build_analysis_prompt(dialog)

    call_kwargs: dict = {
        "model": model,
        "temperature": 0.0,   # deterministic analysis
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }
    # Only pass response_format when the provider supports JSON-object mode
    if cfg.supports_json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}
    if cfg.supports_seed:
        call_kwargs["seed"] = seed

    for attempt in range(3):
        try:
            t0 = time.monotonic()
            response = client.chat.completions.create(**call_kwargs)
            elapsed_ms = round((time.monotonic() - t0) * 1000)
            raw_text = response.choices[0].message.content.strip()

            # Strip markdown fences that some models add despite instructions
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]

            raw_data = json.loads(raw_text)
            result = validate_and_clean(raw_data)
            result["elapsed_ms"] = elapsed_ms
            result["analyzed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            return result

        except (json.JSONDecodeError, KeyError) as exc:
            print(
                f"  [warn] dialog {dialog['id']} attempt {attempt + 1} failed: {exc}",
                file=sys.stderr,
            )
            time.sleep(2 ** attempt)

    print(
        f"  [error] dialog {dialog['id']}: returning default metrics.", file=sys.stderr
    )
    return {
        "intent": "other",
        "satisfaction": "neutral",
        "quality_score": 3,
        "agent_mistakes": [],
        "reasoning": "Analysis failed after 3 attempts.",
        "elapsed_ms": 0,
        "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────

def compute_summary(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {}

    intent_dist: dict[str, int] = {}
    satisfaction_dist: dict[str, int] = {}
    mistake_dist: dict[str, int] = {}
    total_score = 0
    total_elapsed = 0

    # Per-scenario accumulators: {scenario: {score_sum, count, satisfaction counts}}
    scenario_acc: dict[str, dict] = {}

    # Hidden dissatisfaction accuracy
    hidden_total = 0
    hidden_detected = 0  # labeled unsatisfied when ground-truth hidden=True

    for r in results:
        intent_dist[r["intent"]] = intent_dist.get(r["intent"], 0) + 1
        satisfaction_dist[r["satisfaction"]] = (
            satisfaction_dist.get(r["satisfaction"], 0) + 1
        )
        total_score += r["quality_score"]
        total_elapsed += r.get("elapsed_ms", 0)
        for m in r["agent_mistakes"]:
            mistake_dist[m] = mistake_dist.get(m, 0) + 1

        # Scenario breakdown
        sc = r.get("scenario") or "unknown"
        if sc not in scenario_acc:
            scenario_acc[sc] = {"score_sum": 0, "count": 0, "satisfaction": {}}
        scenario_acc[sc]["score_sum"] += r["quality_score"]
        scenario_acc[sc]["count"] += 1
        sat = r["satisfaction"]
        scenario_acc[sc]["satisfaction"][sat] = (
            scenario_acc[sc]["satisfaction"].get(sat, 0) + 1
        )

        # Hidden dissatisfaction accuracy
        if r.get("hidden_dissatisfaction") is True:
            hidden_total += 1
            if r["satisfaction"] == "unsatisfied":
                hidden_detected += 1

    scenario_breakdown = {
        sc: {
            "avg_quality_score": round(acc["score_sum"] / acc["count"], 2),
            "count": acc["count"],
            "satisfaction": acc["satisfaction"],
        }
        for sc, acc in sorted(scenario_acc.items())
    }

    hidden_accuracy: float | None = None
    if hidden_total > 0:
        hidden_accuracy = round(hidden_detected / hidden_total, 2)

    return {
        "total_dialogs": total,
        "avg_quality_score": round(total_score / total, 2),
        "avg_elapsed_ms": round(total_elapsed / total),
        "intent_distribution": intent_dist,
        "satisfaction_distribution": satisfaction_dist,
        "mistake_frequency": dict(
            sorted(mistake_dist.items(), key=lambda x: x[1], reverse=True)
        ),
        "hidden_dissatisfaction_accuracy": {
            "total_hidden_dialogs": hidden_total,
            "detected_as_unsatisfied": hidden_detected,
            "accuracy": hidden_accuracy,
            "note": (
                "Fraction of ground-truth hidden-dissatisfaction dialogs "
                "correctly labelled 'unsatisfied' by the LLM."
            ),
        },
        "scenario_breakdown": scenario_breakdown,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_analysis(
    input_path: Path,
    out_path: Path,
    provider_name: str,
    model: str | None,
    base_seed: int,
    filter_by: str = "all",
) -> None:
    client, cfg = get_client(provider_name)
    effective_model = model or cfg.default_model

    if not input_path.exists():
        sys.exit(
            f"ERROR: Input file '{input_path}' not found.\n"
            "Run generate.py first to create the dataset."
        )

    with open(input_path, encoding="utf-8") as fh:
        dataset: list[dict] = json.load(fh)

    # ── Apply filter ────────────────────────────────────────────────
    if filter_by == "hidden_only":
        dataset = [d for d in dataset if d.get("hidden_dissatisfaction") is True]
        print(f"Filter   : hidden_only → {len(dataset)} dialogs")
    elif filter_by != "all":
        dataset = [d for d in dataset if d.get("scenario") == filter_by]
        print(f"Filter   : scenario={filter_by} → {len(dataset)} dialogs")

    if not dataset:
        sys.exit("ERROR: No dialogs match the given filter. Nothing to analyze.")

    print(
        f"Provider : {cfg.name}\n"
        f"Model    : {effective_model}\n"
        f"Loaded   : {len(dataset)} dialogs from {input_path}\n"
    )

    results = []

    for dialog in dataset:
        seed = base_seed + dialog.get("id", 0)
        print(
            f"[{dialog['id']:>3}/{len(dataset)}] "
            f"scenario={dialog.get('scenario', '?'):<18} "
            f"case={dialog.get('case_type', '?'):<12}"
            f" hidden={str(dialog.get('hidden_dissatisfaction', '?')):<5}",
            end="  ",
            flush=True,
        )
        metrics = analyze_dialog(client, cfg, dialog, effective_model, seed)

        result = {
            "dialog_id":              dialog["id"],
            "scenario":               dialog.get("scenario"),
            "case_type":              dialog.get("case_type"),
            "hidden_dissatisfaction": dialog.get("hidden_dissatisfaction"),
            "sub_scenario":           dialog.get("sub_scenario"),
            "lang":                   dialog.get("lang", "en"),
            **metrics,
        }
        results.append(result)

        print(
            f"intent={result['intent']:<18} "
            f"satisfaction={result['satisfaction']:<12} "
            f"score={result['quality_score']} "
            f"ms={result.get('elapsed_ms', 0):<6} "
            f"mistakes={result['agent_mistakes']}"
        )

    summary = compute_summary(results)

    output = {
        "provider": cfg.name,
        "model":    effective_model,
        "seed":     base_seed,
        "filter":   filter_by,
        "summary":  summary,
        "results":  results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, ensure_ascii=False, indent=2)

    print(f"\n✓ Analysis saved → {out_path}")
    print("\n── Summary ──────────────────────────────────────────")
    # Print a concise version (omit scenario_breakdown for brevity)
    brief = {k: v for k, v in summary.items() if k != "scenario_breakdown"}
    print(json.dumps(brief, ensure_ascii=False, indent=2))
    if "hidden_dissatisfaction_accuracy" in summary:
        hda = summary["hidden_dissatisfaction_accuracy"]
        acc = hda.get("accuracy")
        acc_str = f"{acc:.0%}" if acc is not None else "n/a"
        print(
            f"\nHidden dissatisfaction detection: "
            f"{hda['detected_as_unsatisfied']}/{hda['total_hidden_dialogs']} ({acc_str})"
        )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_provider = auto_detect_provider()
    parser = argparse.ArgumentParser(
        description="Analyze support-chat dialogs with an LLM.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chats.json"),
        help="Input JSON dataset (default: data/chats.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/analysis.json"),
        help="Output JSON path (default: results/analysis.json)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=default_provider,
        choices=list(PROVIDERS.keys()),
        help=(
            "LLM provider (default: auto-detected from env keys)\n"
            "  deepseek   → DEEPSEEK_API_KEY    (model: deepseek-chat)          [free]\n"
            "  qwen       → DASHSCOPE_API_KEY   (model: qwen-turbo)             [free]\n"
            "  groq       → GROQ_API_KEY        (model: llama-3.3-70b-versatile)[free]\n"
            "  gemini     → GEMINI_API_KEY      (model: gemini-2.0-flash)       [free]\n"
            "  mistral    → MISTRAL_API_KEY     (model: mistral-small-latest)   [free]\n"
            "  together   → TOGETHER_API_KEY   (model: Llama-3.3-70B-Free)     [free]\n"
            "  openrouter → OPENROUTER_API_KEY  (model: llama-3.1-8b:free)      [free]\n"
            "  openai     → OPENAI_API_KEY      (model: gpt-4o-mini)            [paid]"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the default model for the chosen provider",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for determinism (default: 42)",
    )
    parser.add_argument(
        "--filter",
        dest="filter_by",
        type=str,
        default="all",
        metavar="FILTER",
        help=(
            "Filter dialogs before analysis (default: all)\n"
            "  all          – analyze every dialog\n"
            "  hidden_only  – only dialogs with hidden_dissatisfaction=true\n"
            "  <scenario>   – e.g. payment_issue, technical_error, refund, ..."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(
        input_path=args.input,
        out_path=args.out,
        provider_name=args.provider,
        model=args.model,
        base_seed=args.seed,
        filter_by=args.filter_by,
    )
