"""
report.py
---------
Reads results/analysis.json (produced by analyze.py) and writes a
human-readable Markdown report to results/report.md.

The report includes:
  - Dataset overview (provider, model, total dialogs, filter applied)
  - Quality score distribution (histogram-style)
  - Satisfaction distribution
  - Intent distribution
  - Most frequent agent mistakes
  - Hidden dissatisfaction detection accuracy
  - Per-scenario breakdown table
  - Top 5 lowest-scoring dialogs (most likely needing attention)
  - Top 5 highest-scoring dialogs (best examples)
  - Full per-dialog detail table

Usage:
    python report.py [--input results/analysis.json] [--out results/report.md]
                     [--title "My Report"]
"""

import argparse
import json
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

BAR_WIDTH = 20   # max width of ASCII bar charts


def bar(value: int, total: int, width: int = BAR_WIDTH) -> str:
    """Return a Unicode progress bar string."""
    if total == 0:
        return ""
    filled = round(value / total * width)
    return "█" * filled + "░" * (width - filled)


def pct(value: int, total: int) -> str:
    if total == 0:
        return "0%"
    return f"{value / total:.0%}"


def score_bar(score: float, max_score: int = 5) -> str:
    """Return a ★ rating string for a quality score."""
    full = round(score)
    return "★" * full + "☆" * (max_score - full)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Build a Markdown table string."""
    sep = ["-" * max(len(h), 3) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Report builder
# ─────────────────────────────────────────────

def build_report(data: dict, title: str) -> str:
    summary = data.get("summary", {})
    results = data.get("results", [])
    provider = data.get("provider", "unknown")
    model = data.get("model", "unknown")
    seed = data.get("seed", "?")
    filter_by = data.get("filter", "all")
    total = summary.get("total_dialogs", len(results))

    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────
    lines += [
        f"# {title}",
        "",
        f"**Provider:** {provider}  ",
        f"**Model:** `{model}`  ",
        f"**Seed:** {seed}  ",
        f"**Filter:** `{filter_by}`  ",
        f"**Total dialogs analyzed:** {total}  ",
        f"**Average quality score:** {summary.get('avg_quality_score', 'n/a')} / 5  ",
        f"**Average LLM latency:** {summary.get('avg_elapsed_ms', 'n/a')} ms  ",
        "",
        "---",
        "",
    ]

    # ── Quality score distribution ────────────────────────────────
    lines += ["## Quality Score Distribution", ""]
    score_counts: dict[int, int] = {i: 0 for i in range(1, 6)}
    for r in results:
        s = r.get("quality_score", 3)
        if isinstance(s, int) and 1 <= s <= 5:
            score_counts[s] += 1

    lines.append("| Score | Count | Bar |")
    lines.append("|-------|-------|-----|")
    for score in range(5, 0, -1):
        count = score_counts[score]
        lines.append(
            f"| {score_bar(score)} ({score}) | {count} ({pct(count, total)}) "
            f"| {bar(count, total)} |"
        )
    lines += ["", "---", ""]

    # ── Satisfaction distribution ─────────────────────────────────
    lines += ["## Satisfaction Distribution", ""]
    sat_dist = summary.get("satisfaction_distribution", {})
    lines.append("| Satisfaction | Count | Bar |")
    lines.append("|-------------|-------|-----|")
    for label in ["satisfied", "neutral", "unsatisfied"]:
        count = sat_dist.get(label, 0)
        lines.append(
            f"| **{label}** | {count} ({pct(count, total)}) | {bar(count, total)} |"
        )
    lines += ["", "---", ""]

    # ── Intent distribution ───────────────────────────────────────
    lines += ["## Intent Distribution", ""]
    intent_dist = summary.get("intent_distribution", {})
    i_rows = [
        [intent, str(count), pct(count, total), bar(count, total)]
        for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)
    ]
    lines.append(md_table(["Intent", "Count", "%", "Bar"], i_rows))
    lines += ["", "---", ""]

    # ── Agent mistake frequency ───────────────────────────────────
    lines += ["## Agent Mistake Frequency", ""]
    mistake_freq = summary.get("mistake_frequency", {})
    if mistake_freq:
        mistake_total = sum(mistake_freq.values())
        m_rows = [
            [mistake, str(count), pct(count, mistake_total)]
            for mistake, count in mistake_freq.items()
        ]
        lines.append(md_table(["Mistake", "Count", "% of all mistakes"], m_rows))
    else:
        lines.append("_No agent mistakes detected._")
    lines += ["", "---", ""]

    # ── Hidden dissatisfaction accuracy ──────────────────────────
    hda = summary.get("hidden_dissatisfaction_accuracy", {})
    if hda:
        lines += ["## Hidden Dissatisfaction Detection", ""]
        acc = hda.get("accuracy")
        acc_str = f"{acc:.0%}" if acc is not None else "n/a"
        lines += [
            f"- **Ground-truth hidden dialogs:** {hda.get('total_hidden_dialogs', 0)}",
            f"- **Correctly labelled `unsatisfied`:** {hda.get('detected_as_unsatisfied', 0)}",
            f"- **Detection accuracy:** **{acc_str}**",
            "",
            "> Hidden dissatisfaction = dialogs where the client politely thanks the agent",
            "> but the underlying problem remains unresolved.",
            "",
            "---",
            "",
        ]

    # ── Per-scenario breakdown ────────────────────────────────────
    scene_breakdown = summary.get("scenario_breakdown", {})
    if scene_breakdown:
        lines += ["## Scenario Breakdown", ""]
        sb_rows = []
        for scenario, info in scene_breakdown.items():
            sat = info.get("satisfaction", {})
            sb_rows.append([
                scenario,
                str(info.get("count", 0)),
                f"{info.get('avg_quality_score', 0):.2f}",
                str(sat.get("satisfied", 0)),
                str(sat.get("neutral", 0)),
                str(sat.get("unsatisfied", 0)),
            ])
        lines.append(md_table(
            ["Scenario", "Count", "Avg Score", "Satisfied", "Neutral", "Unsatisfied"],
            sb_rows,
        ))
        lines += ["", "---", ""]

    # ── Top 5 lowest-scoring dialogs ──────────────────────────────
    sorted_results = sorted(results, key=lambda r: (r.get("quality_score", 3), r.get("dialog_id", 0)))
    lines += ["## 🔴 5 Lowest-Scoring Dialogs (Need Attention)", ""]
    worst = sorted_results[:5]
    w_rows = []
    for r in worst:
        w_rows.append([
            str(r.get("dialog_id")),
            r.get("scenario", "?"),
            r.get("case_type", "?"),
            score_bar(r.get("quality_score", 0)) + f" ({r.get('quality_score', 0)})",
            r.get("satisfaction", "?"),
            ", ".join(r.get("agent_mistakes", [])) or "—",
            r.get("reasoning", "")[:80] + ("…" if len(r.get("reasoning", "")) > 80 else ""),
        ])
    lines.append(md_table(
        ["ID", "Scenario", "Case", "Score", "Satisfaction", "Mistakes", "Reasoning (truncated)"],
        w_rows,
    ))
    lines += ["", "---", ""]

    # ── Top 5 highest-scoring dialogs ────────────────────────────
    best = sorted(results, key=lambda r: (-r.get("quality_score", 3), r.get("dialog_id", 0)))[:5]
    lines += ["## 🟢 5 Highest-Scoring Dialogs (Best Examples)", ""]
    b_rows = []
    for r in best:
        b_rows.append([
            str(r.get("dialog_id")),
            r.get("scenario", "?"),
            r.get("case_type", "?"),
            score_bar(r.get("quality_score", 0)) + f" ({r.get('quality_score', 0)})",
            r.get("satisfaction", "?"),
            r.get("reasoning", "")[:80] + ("…" if len(r.get("reasoning", "")) > 80 else ""),
        ])
    lines.append(md_table(
        ["ID", "Scenario", "Case", "Score", "Satisfaction", "Reasoning (truncated)"],
        b_rows,
    ))
    lines += ["", "---", ""]

    # ── Full per-dialog table ─────────────────────────────────────
    lines += ["## Full Dialog Results", ""]
    all_rows = []
    for r in results:
        hidden_gt = r.get("hidden_dissatisfaction", False)
        hidden_flag = "🔴" if (hidden_gt and r.get("satisfaction") != "unsatisfied") else (
            "✅" if hidden_gt else ""
        )
        all_rows.append([
            str(r.get("dialog_id")),
            r.get("scenario", "?"),
            r.get("case_type", "?"),
            "✓" if hidden_gt else "",
            r.get("intent", "?"),
            r.get("satisfaction", "?"),
            str(r.get("quality_score", "?")),
            ", ".join(r.get("agent_mistakes", [])) or "—",
            hidden_flag,
        ])
    lines.append(md_table(
        ["ID", "Scenario", "Case", "Hidden GT", "Intent", "Satisfaction",
         "Score", "Mistakes", "Missed?"],
        all_rows,
    ))
    lines += [
        "",
        "> **Missed?** 🔴 = ground-truth hidden dissatisfaction NOT detected as `unsatisfied`",
        "> ✅ = correctly detected",
        "",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_report(
    input_path: Path,
    out_path: Path,
    title: str,
) -> None:
    if not input_path.exists():
        sys.exit(
            f"ERROR: Input file '{input_path}' not found.\n"
            "Run analyze.py first to generate the analysis."
        )

    with open(input_path, encoding="utf-8") as fh:
        data = json.load(fh)

    print(f"Loaded   : {input_path}")
    report_md = build_report(data, title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(report_md)

    print(f"✓ Report saved → {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown quality report from analysis results.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/analysis.json"),
        help="Input analysis JSON (default: results/analysis.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/report.md"),
        help="Output Markdown report (default: results/report.md)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Support Chat Quality Report",
        help='Report title (default: "Support Chat Quality Report")',
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_report(
        input_path=args.input,
        out_path=args.out,
        title=args.title,
    )
