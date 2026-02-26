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
                      [--workers N]     (parallel LLM calls, default: 4)
                      [--resume]        (skip already-analyzed dialogs)
                      [--csv PATH]      (also export results to CSV)
"""

import argparse
import csv
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from providers import PROVIDERS, auto_detect_provider, get_client

load_dotenv()

_NO_COLOR = False


def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def green(t: str) -> str:   return _c("32", t)
def yellow(t: str) -> str:  return _c("33", t)
def red(t: str) -> str:     return _c("31", t)
def cyan(t: str) -> str:    return _c("36", t)
def bold(t: str) -> str:    return _c("1",  t)
def dim(t: str) -> str:     return _c("2",  t)

def color_score(score: int) -> str:
    if score >= 4:
        return green(str(score))
    if score == 3:
        return yellow(str(score))
    return red(str(score))

def color_satisfaction(sat: str) -> str:
    mapping: dict = {"satisfied": green, "neutral": yellow, "unsatisfied": red}
    return mapping.get(sat, str)(sat)

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
    seen: set[str] = set()
    agent_mistakes = [m for m in agent_mistakes if not (m in seen or seen.add(m))] 

    reasoning = str(raw.get("reasoning", "")).strip()

    return {
        "intent": intent,
        "satisfaction": satisfaction,
        "quality_score": quality_score,
        "agent_mistakes": agent_mistakes,
        "reasoning": reasoning,
    }

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
        "temperature": 0.0,   
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    }
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
            backoff = 2 ** attempt + random.uniform(0, 0.5)
            did = dialog["id"]
            warn_msg = f"[warn] dialog {did} attempt {attempt + 1} failed: {exc}"
            print(
                f"  {yellow(warn_msg)}  retrying in {backoff:.1f}s",
                file=sys.stderr,
            )
            time.sleep(backoff)

    did = dialog["id"]
    print(
        f"  {red(f'[error] dialog {did}: returning default metrics.')}",
        file=sys.stderr,
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

_print_lock = threading.Lock()

def _analyze_one(args: tuple) -> dict:
    """Worker for ThreadPoolExecutor: wraps analyze_dialog with thread-safe logging."""
    client, cfg, dialog, model, seed, idx, total = args

    with _print_lock:
        print(
            f"  {dim(f'[{idx:>3}/{total}]')} "
            f"scenario={cyan(str(dialog.get('scenario', '?'))):<28} "
            f"case={dialog.get('case_type', '?'):<12} "
            f"hidden={str(dialog.get('hidden_dissatisfaction', '?')):<5}",
            end="  …\r",
            flush=True,
        )

    metrics = analyze_dialog(client, cfg, dialog, model, seed)

    result = {
        "dialog_id":              dialog["id"],
        "scenario":               dialog.get("scenario"),
        "case_type":              dialog.get("case_type"),
        "hidden_dissatisfaction": dialog.get("hidden_dissatisfaction"),
        "sub_scenario":           dialog.get("sub_scenario"),
        "lang":                   dialog.get("lang", "en"),
        **metrics,
    }

    mistakes_str = (
        ", ".join(result["agent_mistakes"]) if result["agent_mistakes"] else dim("—")
    )
    ms_val = result.get("elapsed_ms", 0)
    with _print_lock:
        print(
            f"  {dim(f'[{idx:>3}/{total}]')} "
            f"scenario={cyan(str(result['scenario'])):<28} "
            f"intent={result['intent']:<18} "
            f"sat={color_satisfaction(result['satisfaction']):<22} "
            f"score={color_score(result['quality_score'])} "
            f"{dim(f'ms={ms_val:<6}')} "
            f"[{mistakes_str}]"
        )
    return result


# ─────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────

def export_csv(results: list[dict], csv_path: Path) -> None:
    """Write results to a flat CSV file for easy spreadsheet analysis."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dialog_id", "scenario", "case_type", "hidden_dissatisfaction",
        "sub_scenario", "lang", "intent", "satisfaction", "quality_score",
        "agent_mistakes", "reasoning", "elapsed_ms", "analyzed_at",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = dict(r)
            row["agent_mistakes"] = "|".join(r.get("agent_mistakes", []))
            writer.writerow(row)
    print(f"{green('✓')} CSV  saved → {bold(str(csv_path))}")


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
    workers: int = 4,
    resume: bool = False,
    csv_path: Path | None = None,
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

    # ── Resume: load already-analyzed dialogs ───────────────────────
    existing_results: dict[int, dict] = {}
    if resume and out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as fh:
                saved = json.load(fh)
            for r in saved.get("results", []):
                existing_results[r["dialog_id"]] = r
            print(
                f"Resume   : {dim(str(len(existing_results)))} already-analyzed "
                f"dialog(s) loaded, will skip them."
            )
        except (json.JSONDecodeError, KeyError) as exc:
            print(
                f"{yellow(f'[warn] Could not load existing results for resume: {exc}')}",
                file=sys.stderr,
            )

    pending = [d for d in dataset if d.get("id") not in existing_results]
    skipped = len(dataset) - len(pending)

    print(
        f"\n{bold('Provider')} : {cyan(cfg.name)}\n"
        f"{bold('Model')}    : {cyan(effective_model)}\n"
        f"{bold('Dialogs')}  : {len(dataset)} loaded  "
        f"{green(str(len(pending)))} to analyze  {dim(str(skipped) + ' skipped')}\n"
        f"{bold('Workers')}  : {workers} parallel thread(s)\n"
        f"{bold('Input')}    : {input_path}\n"
    )

    # ── Parallel analysis ────────────────────────────────────────────
    new_results: list[dict] = []
    wall_start = time.monotonic()

    if workers == 1 or len(pending) <= 1:
        for idx, dialog in enumerate(pending, start=skipped + 1):
            seed = base_seed + dialog.get("id", 0)
            new_results.append(
                _analyze_one((client, cfg, dialog, effective_model, seed, idx, len(dataset)))
            )
    else:
        work_items = [
            (client, cfg, d, effective_model, base_seed + d.get("id", 0), idx, len(dataset))
            for idx, d in enumerate(pending, start=skipped + 1)
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_analyze_one, item): item for item in work_items}
            for future in as_completed(futures):
                try:
                    new_results.append(future.result())
                except Exception as exc:
                    with _print_lock:
                        print(f"{red(f'[error] worker failed: {exc}')}", file=sys.stderr)

    wall_elapsed = round((time.monotonic() - wall_start) * 1000)

    # Merge new + previously-skipped, sort by dialog_id
    results = sorted(
        list(existing_results.values()) + new_results,
        key=lambda r: r["dialog_id"],
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

    if csv_path:
        export_csv(results, csv_path)

    print(f"\n{green('✓')} Analysis saved → {bold(str(out_path))}")
    if pending:
        per_dialog = round(wall_elapsed / len(pending))
        print(
            f"  Wall time : {wall_elapsed} ms for {len(pending)} dialog(s) "
            f"({per_dialog} ms/dialog avg)"
        )

    print(f"\n{bold('── Summary ──────────────────────────────────────────')}")
    brief = {k: v for k, v in summary.items() if k != "scenario_breakdown"}
    print(json.dumps(brief, ensure_ascii=False, indent=2))

    if "hidden_dissatisfaction_accuracy" in summary:
        hda = summary["hidden_dissatisfaction_accuracy"]
        acc = hda.get("accuracy")
        acc_str = f"{acc:.0%}" if acc is not None else "n/a"
        color_fn = green if (acc or 0) >= 0.7 else (yellow if (acc or 0) >= 0.4 else red)
        print(
            f"\n{bold('Hidden dissatisfaction detection:')} "
            f"{hda['detected_as_unsatisfied']}/{hda['total_hidden_dialogs']} "
            f"({color_fn(acc_str)})"
        )

    if summary.get("scenario_breakdown"):
        print(f"\n{bold('── Scenario Breakdown ───────────────────────────────')}")
        print(f"  {'Scenario':<22} {'Avg':>5}  Sat  Neu  Unsat")
        print(f"  {'-'*22} {'-'*5}  ---  ---  -----")
        for sc, info in summary["scenario_breakdown"].items():
            sat_d = info["satisfaction"]
            avg = info["avg_quality_score"]
            avg_c = (
                green(f"{avg:.2f}") if avg >= 4
                else (yellow(f"{avg:.2f}") if avg >= 3 else red(f"{avg:.2f}"))
            )
            print(
                f"  {sc:<22} {avg_c}  "
                f"{green(str(sat_d.get('satisfied', 0))):>4}  "
                f"{yellow(str(sat_d.get('neutral', 0))):>4}  "
                f"{red(str(sat_d.get('unsatisfied', 0))):>5}"
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
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        metavar="N",
        help=(
            "Parallel LLM calls (default: 4).\n"
            "Set to 1 to disable concurrency (sequential, easier to debug)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip dialogs already present in --out (saves API calls after interruption).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Also export results to a flat CSV file (e.g. results/analysis.csv).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable ANSI color output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_color:
        _NO_COLOR = True
    run_analysis(
        input_path=args.input,
        out_path=args.out,
        provider_name=args.provider,
        model=args.model,
        base_seed=args.seed,
        filter_by=args.filter_by,
        workers=args.workers,
        resume=args.resume,
        csv_path=args.csv,
    )
