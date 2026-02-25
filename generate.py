"""
generate.py
-----------
Generates a deterministic dataset of support chat dialogs using an LLM.

Each dialog is tagged with:
  - scenario : payment_issue | technical_error | account_access | tariff_question | refund
  - case_type : successful | problematic | conflict | agent_error
  - hidden_dissatisfaction : bool  (client politely thanks but issue unresolved)

Usage:
    python generate.py [--count N] [--out data/chats.json]
                       [--provider deepseek|qwen|openai] [--model MODEL]

Default provider: auto-detected from available env keys (deepseek → qwen → openai)
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from providers import PROVIDERS, auto_detect_provider, get_client

load_dotenv()

# ─────────────────────────────────────────────
# Scenario matrix: (scenario, case_type, hidden_dissatisfaction)
# ─────────────────────────────────────────────
SCENARIOS = [
    # ── payment_issue ──────────────────────────────────────────────
    ("payment_issue",    "successful",    False),
    ("payment_issue",    "problematic",   False),
    ("payment_issue",    "conflict",      False),
    ("payment_issue",    "agent_error",   False),
    ("payment_issue",    "problematic",   True),   # hidden dissatisfaction

    # ── technical_error ────────────────────────────────────────────
    ("technical_error",  "successful",    False),
    ("technical_error",  "problematic",   False),
    ("technical_error",  "conflict",      False),
    ("technical_error",  "agent_error",   False),
    ("technical_error",  "problematic",   True),

    # ── account_access ─────────────────────────────────────────────
    ("account_access",   "successful",    False),
    ("account_access",   "problematic",   False),
    ("account_access",   "conflict",      False),
    ("account_access",   "agent_error",   False),
    ("account_access",   "successful",    True),   # hidden: thanked but still locked

    # ── tariff_question ────────────────────────────────────────────
    ("tariff_question",  "successful",    False),
    ("tariff_question",  "problematic",   False),
    ("tariff_question",  "conflict",      False),
    ("tariff_question",  "agent_error",   False),
    ("tariff_question",  "problematic",   True),

    # ── refund ─────────────────────────────────────────────────────
    ("refund",           "successful",    False),
    ("refund",           "problematic",   False),
    ("refund",           "conflict",      False),
    ("refund",           "agent_error",   False),
    ("refund",           "conflict",      True),   # paid lip-service, refund denied
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


def build_prompt(scenario: str, case_type: str, hidden: bool) -> str:
    hidden_note = HIDDEN_DISSATISFACTION_NOTE if hidden else ""
    return f"""You are generating a realistic customer-support chat for a SaaS product.

Topic: {SCENARIO_DESC[scenario]}
Case type: {CASE_DESC[case_type]}
{hidden_note}
Rules:
1. Write 6–14 turns alternating between CLIENT and AGENT.
2. Make the language natural, slightly informal but professional.
3. Include realistic details (order IDs, dates, error codes, plan names, amounts).
4. Return ONLY a JSON object with this exact schema – no markdown fences, no extra keys:

{{
  "messages": [
    {{"role": "client", "text": "..."}},
    {{"role": "agent",  "text": "..."}}
  ]
}}

The first message must be from the client.
"""


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

def generate_dialog(
    client,
    cfg,
    scenario: str,
    case_type: str,
    hidden: bool,
    model: str,
    seed: int,
) -> list[dict]:
    """Call the LLM and return the parsed messages list."""
    prompt = build_prompt(scenario, case_type, hidden)

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
            if not messages:
                raise ValueError("Empty messages list returned by LLM.")
            return messages

        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  [warn] attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to generate dialog after 3 attempts: {scenario}/{case_type}")


def run_generation(
    count: int,
    out_path: Path,
    provider_name: str,
    model: str | None,
    base_seed: int = 42,
) -> None:
    client, cfg = get_client(provider_name)
    effective_model = model or cfg.default_model

    print(
        f"Provider : {cfg.name}\n"
        f"Model    : {effective_model}\n"
        f"Dialogs  : {count}\n"
        f"Output   : {out_path}\n"
    )

    # Cycle through the scenario matrix to reach the requested count
    matrix = SCENARIOS * (count // len(SCENARIOS) + 1)
    matrix = matrix[:count]

    dataset = []
    for idx, (scenario, case_type, hidden) in enumerate(matrix):
        seed = base_seed + idx
        print(
            f"[{idx + 1:>3}/{count}] scenario={scenario:<18} "
            f"case={case_type:<12} hidden={str(hidden):<5} seed={seed}"
        )
        messages = generate_dialog(client, cfg, scenario, case_type, hidden, effective_model, seed)

        dataset.append(
            {
                "id": idx + 1,
                "scenario": scenario,
                "case_type": case_type,
                "hidden_dissatisfaction": hidden,
                "seed": seed,
                "messages": messages,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dataset, fh, ensure_ascii=False, indent=2)

    print(f"\n✓ Dataset saved → {out_path}  ({len(dataset)} dialogs)")


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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_generation(
        count=args.count,
        out_path=args.out,
        provider_name=args.provider,
        model=args.model,
        base_seed=args.seed,
    )
