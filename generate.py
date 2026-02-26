"""
generate.py
-----------
Generates a deterministic dataset of support chat dialogs using an LLM.

Each dialog is tagged with:
  - scenario              : payment_issue | technical_error | account_access |
                            tariff_question | refund
  - case_type             : successful | problematic | conflict | agent_error
  - hidden_dissatisfaction: bool  (client politely thanks but issue unresolved)
  - lang                  : en | uk  (language the dialog was generated in)
  - generated_at          : ISO-8601 UTC timestamp
  - sub_scenario          : fine-grained variant hint used in the prompt

Features:
  --resume    Skip dialogs whose id already exists in the output file.
  --lang      en (default) | uk | mixed (alternates en/uk by index).
  --count N   Total dialogs in the output file.

Usage:
    python generate.py [--count N] [--out data/chats.json]
                       [--provider PROVIDER] [--model MODEL]
                       [--seed N] [--lang en|uk|mixed] [--resume]

Default provider: auto-detected from available env keys
"""

import argparse
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

_print_lock = threading.Lock()
_save_lock  = threading.Lock()

# ─────────────────────────────────────────────
# Scenario matrix: (scenario, case_type, hidden_dissatisfaction, sub_scenario)
# ─────────────────────────────────────────────
SCENARIOS: list[tuple[str, str, bool, str]] = [
    # ── payment_issue ──────────────────────────────────────────────
    ("payment_issue",    "successful",    False, "duplicate charge on credit card"),
    ("payment_issue",    "problematic",   False, "payment declined at checkout"),
    ("payment_issue",    "conflict",      False, "wrong amount charged, client escalates"),
    ("payment_issue",    "agent_error",   False, "agent gives wrong refund timeline"),
    ("payment_issue",    "problematic",   True,  "subscription renewed unexpectedly, hidden frustration"),

    # ── technical_error ────────────────────────────────────────────
    ("technical_error",  "successful",    False, "app crashes on login after update"),
    ("technical_error",  "problematic",   False, "feature broken for 3 days, no ETA"),
    ("technical_error",  "conflict",      False, "data lost after server migration"),
    ("technical_error",  "agent_error",   False, "agent blames client's device incorrectly"),
    ("technical_error",  "problematic",   True,  "intermittent 500 errors, client accepts workaround reluctantly"),

    # ── account_access ─────────────────────────────────────────────
    ("account_access",   "successful",    False, "2FA loop preventing login"),
    ("account_access",   "problematic",   False, "account locked after failed attempts"),
    ("account_access",   "conflict",      False, "account suspended without notice"),
    ("account_access",   "agent_error",   False, "agent sends reset link to wrong email"),
    ("account_access",   "successful",    True,  "password reset done but sessions still invalid"),

    # ── tariff_question ────────────────────────────────────────────
    ("tariff_question",  "successful",    False, "comparing Pro vs Business plan features"),
    ("tariff_question",  "problematic",   False, "unclear what is included in current plan"),
    ("tariff_question",  "conflict",      False, "price increased without prior notice"),
    ("tariff_question",  "agent_error",   False, "agent quotes incorrect pricing, not corrected"),
    ("tariff_question",  "problematic",   True,  "downgrade request acknowledged but not processed"),

    # ── refund ─────────────────────────────────────────────────────
    ("refund",           "successful",    False, "refund for accidental annual subscription"),
    ("refund",           "problematic",   False, "refund outside policy window, partial offered"),
    ("refund",           "conflict",      False, "client demands refund, policy dispute"),
    ("refund",           "agent_error",   False, "agent promises refund but it never arrives"),
    ("refund",           "conflict",      True,  "refund denied but client accepts politely out of exhaustion"),
]

# ─────────────────────────────────────────────
# Prompt helpers
# ─────────────────────────────────────────────

SCENARIO_DESC = {
    "payment_issue":   "a payment processing problem (duplicate charge, declined card, wrong amount, etc.)",
    "technical_error": "a technical bug or service outage (app crash, feature not working, 500 error, etc.)",
    "account_access":  "an account access issue (forgotten password, 2FA loop, account locked, etc.)",
    "tariff_question": "a question about subscription plans, pricing, or feature availability",
    "refund":          "a refund or chargeback request for a recent transaction",
}

CASE_DESC = {
    "successful":  "The agent resolves the issue fully, is polite and professional.",
    "problematic": (
        "The issue is difficult to solve. The agent tries but the resolution is incomplete "
        "or requires the client to wait. Client remains somewhat frustrated."
    ),
    "conflict": (
        "The client becomes increasingly angry. The agent either escalates tension "
        "or is passive-aggressive. The conversation ends without full resolution."
    ),
    "agent_error": (
        "The agent makes one or more clear mistakes: gives wrong information, "
        "ignores a direct question, is rude, or escalates unnecessarily."
    ),
}

HIDDEN_DISSATISFACTION_NOTE = """
IMPORTANT – hidden dissatisfaction:
The client must APPEAR satisfied at the end (says "thank you", "okay", "I understand").
However the underlying problem is NOT actually resolved.
The client's final messages should subtly hint at remaining frustration
(e.g. slightly terse replies, vague acceptance, no genuine enthusiasm).
Do NOT make it obvious – a careful analyst should be able to catch it.
"""

LANG_INSTRUCTION = {
    "en": "Write the entire conversation in English.",
    "uk": (
        "Напишіть весь діалог українською мовою. "
        "Використовуйте природню розмовну українську з елементами ділового стилю."
    ),
}


def build_prompt(scenario: str, case_type: str, hidden: bool, sub_scenario: str, lang: str) -> str:
    hidden_note = HIDDEN_DISSATISFACTION_NOTE if hidden else ""
    lang_instr = LANG_INSTRUCTION.get(lang, LANG_INSTRUCTION["en"])
    return f"""You are generating a realistic customer-support chat for a SaaS product.

Language instruction: {lang_instr}

Topic: {SCENARIO_DESC[scenario]}
Specific sub-scenario: {sub_scenario}
Case type: {CASE_DESC[case_type]}
{hidden_note}
Rules:
1. Write 6–14 turns alternating between CLIENT and AGENT.
2. Make the language natural, slightly informal but professional.
3. Include realistic details (order IDs, dates, error codes, plan names, amounts).
4. The sub-scenario description above must be the core of the conversation.
5. Return ONLY a JSON object with this exact schema – no markdown fences, no extra keys:

{{
  "messages": [
    {{"role": "client", "text": "..."}},
    {{"role": "agent",  "text": "..."}}
  ]
}}

The first message must be from the client.
"""


# ─────────────────────────────────────────────
# Validation helpers
# ─────────────────────────────────────────────

def validate_messages(messages: list[dict]) -> list[dict]:
    """
    Ensure messages alternate roles (client/agent) and start with client.
    Filters out any entries missing 'role' or 'text'.
    Raises ValueError if the result is empty or roles do not alternate.
    """
    cleaned = [
        m for m in messages
        if isinstance(m, dict) and m.get("role") in {"client", "agent"} and m.get("text")
    ]
    if not cleaned:
        raise ValueError("No valid messages after filtering.")
    if cleaned[0]["role"] != "client":
        raise ValueError("First message must be from client.")
    for i in range(1, len(cleaned)):
        if cleaned[i]["role"] == cleaned[i - 1]["role"]:
            raise ValueError(f"Consecutive same-role messages at index {i}.")
    return cleaned


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

def generate_dialog(
    client,
    cfg,
    scenario: str,
    case_type: str,
    hidden: bool,
    sub_scenario: str,
    lang: str,
    model: str,
    seed: int,
) -> list[dict]:
    """Call the LLM and return the validated messages list."""
    prompt = build_prompt(scenario, case_type, hidden, sub_scenario, lang)

    call_kwargs: dict = {
        "model": model,
        "temperature": 0.7,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a dataset-generation assistant. "
                    "Always respond with valid JSON only – no markdown fences, no extra keys."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }
    # Only pass response_format when the provider supports it
    if cfg.supports_json_mode:
        call_kwargs["response_format"] = {"type": "json_object"}

    # Some providers support deterministic seed; others ignore it gracefully
    if cfg.supports_seed:
        call_kwargs["seed"] = seed

    last_exc: Exception = RuntimeError("unknown")
    for attempt in range(3):
        try:
            response = client.chat.completions.create(**call_kwargs)
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences that some models add despite instructions
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw)
            messages = data.get("messages", [])
            return validate_messages(messages)

        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            backoff = 2 ** attempt + random.uniform(0, 0.5)
            print(f"  [warn] attempt {attempt + 1} failed: {exc} (retrying in {backoff:.1f}s)", file=sys.stderr)
            time.sleep(backoff)

    raise RuntimeError(
        f"Failed to generate dialog after 3 attempts ({scenario}/{case_type}): {last_exc}"
    )


# ─────────────────────────────────────────────
# Thread-safe parallel worker
# ─────────────────────────────────────────────

def _generate_one(args: tuple) -> dict:
    """Worker for ThreadPoolExecutor. Returns a completed dialog entry."""
    (
        client, cfg, dialog_id, scenario, case_type, hidden,
        sub_scenario, dialog_lang, effective_model, seed, count,
    ) = args

    with _print_lock:
        print(
            f"[{dialog_id:>3}/{count}] GEN   scenario={scenario:<18} "
            f"case={case_type:<12} hidden={str(hidden):<5} "
            f"lang={dialog_lang} seed={seed}",
            end="  …\r",
            flush=True,
        )

    t0 = time.monotonic()
    messages = generate_dialog(
        client, cfg, scenario, case_type, hidden, sub_scenario,
        dialog_lang, effective_model, seed,
    )
    elapsed = round((time.monotonic() - t0) * 1000)

    with _print_lock:
        print(
            f"[{dialog_id:>3}/{count}] GEN   scenario={scenario:<18} "
            f"case={case_type:<12} hidden={str(hidden):<5} "
            f"lang={dialog_lang}  turns={len(messages)} ({elapsed} ms)"
        )

    return {
        "id":                    dialog_id,
        "scenario":              scenario,
        "case_type":             case_type,
        "hidden_dissatisfaction": hidden,
        "sub_scenario":          sub_scenario,
        "lang":                  dialog_lang,
        "seed":                  seed,
        "generated_at":          datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "messages":              messages,
    }


def run_generation(
    count: int,
    out_path: Path,
    provider_name: str,
    model: str | None,
    base_seed: int = 42,
    lang: str = "en",
    resume: bool = False,
    workers: int = 3,
) -> None:
    client, cfg = get_client(provider_name)
    effective_model = model or cfg.default_model

    # ── Resume: load existing dataset ──────────────────────────────
    existing: dict[int, dict] = {}
    if resume and out_path.exists():
        try:
            with open(out_path, encoding="utf-8") as fh:
                for entry in json.load(fh):
                    existing[entry["id"]] = entry
            print(f"Resume: found {len(existing)} existing dialog(s), will skip them.\n")
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"[warn] Could not load existing file for resume: {exc}", file=sys.stderr)

    print(
        f"Provider : {cfg.name}\n"
        f"Model    : {effective_model}\n"
        f"Dialogs  : {count}\n"
        f"Language : {lang}\n"
        f"Workers  : {workers} parallel thread(s)\n"
        f"Output   : {out_path}\n"
    )

    # Cycle through the scenario matrix to reach the requested count
    matrix = SCENARIOS * (count // len(SCENARIOS) + 1)
    matrix = matrix[:count]

    # Partition work: skipped vs. to-generate
    skipped_entries: list[dict] = []
    work_items: list[tuple] = []

    for idx, (scenario, case_type, hidden, sub_scenario) in enumerate(matrix):
        dialog_id = idx + 1
        seed = base_seed + idx

        if lang == "mixed":
            dialog_lang = "en" if idx % 2 == 0 else "uk"
        else:
            dialog_lang = lang

        if dialog_id in existing:
            skipped_entries.append(existing[dialog_id])
            print(
                f"[{dialog_id:>3}/{count}] SKIP  scenario={scenario:<18} "
                f"case={case_type:<12} (already exists)"
            )
        else:
            work_items.append((
                client, cfg, dialog_id, scenario, case_type, hidden,
                sub_scenario, dialog_lang, effective_model, seed, count,
            ))

    # ── Generate (parallel) ─────────────────────────────────────────
    new_entries: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _save_current() -> None:
        all_entries = sorted(
            skipped_entries + new_entries,
            key=lambda e: e["id"],
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(all_entries, fh, ensure_ascii=False, indent=2)

    if workers == 1 or len(work_items) <= 1:
        for item in work_items:
            entry = _generate_one(item)
            new_entries.append(entry)
            _save_current()   # incremental save
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_generate_one, item): item for item in work_items}
            for future in as_completed(futures):
                try:
                    entry = future.result()
                    with _save_lock:
                        new_entries.append(entry)
                        _save_current()     # incremental thread-safe save
                except Exception as exc:
                    with _print_lock:
                        print(f"[error] worker failed: {exc}", file=sys.stderr)

    total_entries = len(skipped_entries) + len(new_entries)
    print(
        f"\n✓ Dataset saved → {out_path}  "
        f"({total_entries} dialogs total, "
        f"{len(new_entries)} newly generated, "
        f"{len(skipped_entries)} skipped)"
    )


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    default_provider = auto_detect_provider()
    parser = argparse.ArgumentParser(
        description="Generate a support-chat dataset using an LLM.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--count",
        type=int,
        default=len(SCENARIOS),
        help=f"Number of dialogs to generate (default: {len(SCENARIOS)})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/chats.json"),
        help="Output JSON file path (default: data/chats.json)",
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
        "--lang",
        type=str,
        default="en",
        choices=["en", "uk", "mixed"],
        help=(
            "Language for generated dialogs (default: en)\n"
            "  en    – English only\n"
            "  uk    – Ukrainian only\n"
            "  mixed – alternates en/uk by dialog index"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip dialogs already present in the output file (useful after interruption)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Number of parallel LLM generation calls (default: 3).\n"
            "Set to 1 for sequential generation (easier to debug)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generation(
        count=args.count,
        out_path=args.out,
        provider_name=args.provider,
        model=args.model,
        base_seed=args.seed,
        lang=args.lang,
        resume=args.resume,
        workers=args.workers,
    )
