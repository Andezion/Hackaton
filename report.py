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

BAR_WIDTH = 20  

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

def build_report(data: dict, title: str) -> str:
    summary = data.get("summary", {})
    results = data.get("results", [])
    provider = data.get("provider", "unknown")
    model = data.get("model", "unknown")
    seed = data.get("seed", "?")
    filter_by = data.get("filter", "all")
    total = summary.get("total_dialogs", len(results))

    lines: list[str] = []

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

    lines += ["## Intent Distribution", ""]
    intent_dist = summary.get("intent_distribution", {})
    i_rows = [
        [intent, str(count), pct(count, total), bar(count, total)]
        for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)
    ]
    lines.append(md_table(["Intent", "Count", "%", "Bar"], i_rows))
    lines += ["", "---", ""]

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

def build_html_report(data: dict, title: str) -> str:
    """Build a fully self-contained interactive HTML report with Chart.js."""
    import json as _json

    summary  = data.get("summary", {})
    results  = data.get("results", [])
    provider = data.get("provider", "unknown")
    model    = data.get("model",    "unknown")
    total    = summary.get("total_dialogs", len(results))
    avg_qs   = summary.get("avg_quality_score", 0)
    avg_ms   = summary.get("avg_elapsed_ms", 0)
    filter_v = data.get("filter", "all")

    score_counts = {i: 0 for i in range(1, 6)}
    for r in results:
        s = r.get("quality_score", 3)
        if isinstance(s, int) and 1 <= s <= 5:
            score_counts[s] += 1

    sat_dist     = summary.get("satisfaction_distribution", {})
    sat_labels   = ["satisfied", "neutral", "unsatisfied"]
    sat_values   = [sat_dist.get(lbl, 0) for lbl in sat_labels]

    intent_dist    = summary.get("intent_distribution", {})
    sorted_intents = sorted(intent_dist.items(), key=lambda x: x[1], reverse=True)
    intent_labels  = [k for k, _ in sorted_intents]
    intent_values  = [v for _, v in sorted_intents]

    mistake_freq   = summary.get("mistake_frequency", {})
    mistake_labels = list(mistake_freq.keys())
    mistake_values = list(mistake_freq.values())

    scene_breakdown = summary.get("scenario_breakdown", {})

    hda          = summary.get("hidden_dissatisfaction_accuracy", {})
    hda_total    = hda.get("total_hidden_dialogs", 0)
    hda_detected = hda.get("detected_as_unsatisfied", 0)
    hda_missed   = hda_total - hda_detected
    hda_acc      = hda.get("accuracy")
    hda_acc_str  = f"{hda_acc:.0%}" if hda_acc is not None else "n/a"
    hda_cls      = "good" if (hda_acc or 0) >= 0.7 else ("warn" if (hda_acc or 0) >= 0.4 else "bad")
    qs_cls       = "good" if avg_qs >= 4 else ("warn" if avg_qs >= 3 else "bad")

    table_rows: list[str] = []
    for r in results:
        score     = r.get("quality_score", 0)
        sat       = r.get("satisfaction", "")
        hidden_gt = r.get("hidden_dissatisfaction", False)
        missed    = hidden_gt and sat != "unsatisfied"
        score_cls = "good" if score >= 4 else ("warn" if score == 3 else "bad")
        sat_cls   = {"satisfied": "good", "neutral": "warn", "unsatisfied": "bad"}.get(sat, "")
        mistakes  = ", ".join(r.get("agent_mistakes", [])) or "\u2014"
        reasoning = r.get("reasoning", "").replace('"', "&quot;").replace("<", "&lt;")
        table_rows.append(
            f'<tr data-score="{score}" data-sat="{sat}" data-scenario="{r.get("scenario","")}">'
            f'<td>{r.get("dialog_id")}</td>'
            f'<td>{r.get("scenario","")}</td>'
            f'<td>{r.get("case_type","")}</td>'
            f'<td style="text-align:center">{"🔒" if hidden_gt else ""}</td>'
            f'<td>{r.get("intent","")}</td>'
            f'<td class="{sat_cls}">{sat}</td>'
            f'<td class="{score_cls}" style="text-align:center">{score}</td>'
            f'<td>{mistakes}</td>'
            f'<td style="text-align:center">{"🔴" if missed else ("✅" if hidden_gt else "")}</td>'
            f'<td class="reasoning" title="{reasoning}">'
            f'{reasoning[:90]}{"…" if len(r.get("reasoning","")) > 90 else ""}</td>'
            f'</tr>'
        )

    sc_rows: list[str] = []
    for sc, info in sorted(scene_breakdown.items()):
        avg   = info.get("avg_quality_score", 0)
        cnt   = info.get("count", 0)
        sat_d = info.get("satisfaction", {})
        bar_pct = int(avg / 5 * 100)
        bar_cls = "good" if avg >= 4 else ("warn" if avg >= 3 else "bad")
        sc_rows.append(
            f"<tr><td>{sc}</td><td>{cnt}</td>"
            f'<td><div class="mini-bar">'
            f'<div class="mini-fill {bar_cls}" style="width:{bar_pct}%">{avg:.2f}</div>'
            f'</div></td>'
            f'<td class="good">{sat_d.get("satisfied", 0)}</td>'
            f'<td class="warn">{sat_d.get("neutral", 0)}</td>'
            f'<td class="bad">{sat_d.get("unsatisfied", 0)}</td></tr>'
        )

    mistakes_js_block = ""
    if mistake_labels:
        mistakes_js_block = f"""
new Chart(document.getElementById('mistakeChart'), {{
  type: 'bar',
  data: {{
    labels: {_json.dumps(mistake_labels)},
    datasets: [{{ data: {_json.dumps(mistake_values)},
      backgroundColor: MISTAKE_COLORS, borderRadius: 6, borderSkipped: false }}]
  }},
  options: {{
    indexAxis: 'y', responsive: true,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ beginAtZero: true, ticks: {{ stepSize: 1 }} }},
               y: {{ grid: {{ display: false }} }} }}
  }}
}});"""

    hda_js_block = ""
    if hda_total > 0:
        hda_js_block = f"""
new Chart(document.getElementById('hdaChart'), {{
  type: 'doughnut',
  data: {{
    labels: ['Detected', 'Missed'],
    datasets: [{{ data: [{hda_detected}, {hda_missed}],
      backgroundColor: ['#22c55e','#ef4444'], borderColor: '#1a1d27', borderWidth: 3 }}]
  }},
  options: {{
    responsive: true, cutout: '60%',
    plugins: {{
      legend: {{ position: 'bottom', labels: {{ padding: 16 }} }},
      tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.label}}: ${{ctx.raw}}` }} }}
    }}
  }}
}});"""

    mistake_card_inner = (
        '<canvas id="mistakeChart"></canvas>' if mistake_labels
        else '<p style="color:var(--muted);margin-top:40px;text-align:center">No mistakes recorded ✅</p>'
    )
    hda_card_inner = (
        '<canvas id="hdaChart"></canvas>' if hda_total > 0
        else '<p style="color:var(--muted);margin-top:40px;text-align:center">No hidden dialogs in dataset</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
:root {{
  --bg:#0f1117;--surface:#1a1d27;--card:#21253a;--border:#2e3352;
  --text:#e2e8f0;--muted:#94a3b8;
  --good:#22c55e;--warn:#f59e0b;--bad:#ef4444;
  --accent:#6366f1;--accent2:#38bdf8;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,sans-serif;font-size:14px}}
header{{background:linear-gradient(135deg,#1e1b4b,#0f172a);padding:28px 32px;border-bottom:1px solid var(--border)}}
header h1{{font-size:1.6rem;font-weight:700;color:#a5b4fc;margin-bottom:6px}}
header .meta{{color:var(--muted);font-size:.85rem}}
header .meta span{{margin-right:20px}}
.kpi-row{{display:flex;gap:16px;padding:20px 32px;flex-wrap:wrap}}
.kpi{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 24px;min-width:140px;flex:1}}
.kpi .label{{font-size:.75rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
.kpi .value{{font-size:2rem;font-weight:700;color:var(--accent);margin-top:4px}}
.kpi .value.good{{color:var(--good)}}.kpi .value.warn{{color:var(--warn)}}.kpi .value.bad{{color:var(--bad)}}
main{{padding:0 32px 48px}}
h2{{font-size:1.05rem;font-weight:600;color:#a5b4fc;margin:28px 0 14px;border-left:3px solid var(--accent);padding-left:10px}}
.charts-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}}
.chart-card{{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:20px}}
.chart-card h3{{font-size:.8rem;color:var(--muted);text-transform:uppercase;margin-bottom:14px;letter-spacing:.05em}}
canvas{{max-height:220px}}
.table-wrap{{overflow-x:auto;border-radius:12px;border:1px solid var(--border);margin-top:4px}}
table{{width:100%;border-collapse:collapse}}
thead tr{{background:var(--surface)}}
th{{padding:10px 12px;text-align:left;font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;white-space:nowrap}}
td{{padding:9px 12px;border-top:1px solid var(--border);font-size:.82rem;vertical-align:top}}
tr:hover td{{background:rgba(99,102,241,.06)}}
td.good{{color:var(--good);font-weight:600}}td.warn{{color:var(--warn);font-weight:600}}td.bad{{color:var(--bad);font-weight:600}}
td.reasoning{{color:var(--muted);font-size:.78rem;max-width:260px}}
.mini-bar{{background:var(--border);border-radius:6px;height:22px;overflow:hidden}}
.mini-fill{{height:100%;display:flex;align-items:center;padding-left:8px;font-size:.78rem;font-weight:600;border-radius:6px;color:#fff}}
.mini-fill.good{{background:var(--good)}}.mini-fill.warn{{background:var(--warn);color:#000}}.mini-fill.bad{{background:var(--bad)}}
.filter-bar{{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap;align-items:center}}
.filter-bar input,.filter-bar select{{background:var(--card);border:1px solid var(--border);color:var(--text);padding:6px 12px;border-radius:8px;font-size:.83rem;outline:none}}
.filter-bar input:focus,.filter-bar select:focus{{border-color:var(--accent)}}
footer{{text-align:center;color:var(--muted);font-size:.75rem;padding:24px;border-top:1px solid var(--border)}}
</style>
</head>
<body>

<header>
  <h1>📊 {title}</h1>
  <div class="meta">
    <span>🤖 {provider} / <code>{model}</code></span>
    <span>🔍 Filter: <code>{filter_v}</code></span>
    <span>📅 <span id="dt"></span></span>
  </div>
</header>
<script>document.getElementById('dt').textContent=new Date().toLocaleDateString('en-GB');</script>

<div class="kpi-row">
  <div class="kpi"><div class="label">Total Dialogs</div><div class="value">{total}</div></div>
  <div class="kpi"><div class="label">Avg Quality Score</div>
    <div class="value {qs_cls}">{avg_qs}<span style="font-size:1rem"> /5</span></div></div>
  <div class="kpi"><div class="label">Satisfied</div>
    <div class="value good">{sat_dist.get("satisfied", 0)}</div></div>
  <div class="kpi"><div class="label">Unsatisfied</div>
    <div class="value bad">{sat_dist.get("unsatisfied", 0)}</div></div>
  <div class="kpi"><div class="label">Avg Latency</div>
    <div class="value">{avg_ms}<span style="font-size:1rem"> ms</span></div></div>
  <div class="kpi"><div class="label">Hidden Detection</div>
    <div class="value {hda_cls}">{hda_acc_str}</div></div>
</div>

<main>

<h2>Charts</h2>
<div class="charts-grid">
  <div class="chart-card"><h3>Quality Score Distribution</h3><canvas id="scoreChart"></canvas></div>
  <div class="chart-card"><h3>Customer Satisfaction</h3><canvas id="satChart"></canvas></div>
  <div class="chart-card"><h3>Intent Distribution</h3><canvas id="intentChart"></canvas></div>
  <div class="chart-card"><h3>Agent Mistake Frequency</h3>{mistake_card_inner}</div>
  <div class="chart-card"><h3>Hidden Dissatisfaction Detection</h3>{hda_card_inner}</div>
</div>

<h2>Scenario Breakdown</h2>
<div class="table-wrap">
  <table>
    <thead><tr>
      <th>Scenario</th><th>Count</th><th>Avg Quality</th>
      <th style="color:var(--good)">Satisfied</th>
      <th style="color:var(--warn)">Neutral</th>
      <th style="color:var(--bad)">Unsatisfied</th>
    </tr></thead>
    <tbody>{"".join(sc_rows)}</tbody>
  </table>
</div>

<h2>Dialog Results</h2>
<div class="filter-bar">
  <input id="search" type="text" placeholder="Search scenario / intent / reasoning…" style="min-width:240px" />
  <select id="filterSat">
    <option value="">All satisfaction</option>
    <option value="satisfied">satisfied</option>
    <option value="neutral">neutral</option>
    <option value="unsatisfied">unsatisfied</option>
  </select>
  <select id="filterScore">
    <option value="">All scores</option>
    <option value="5">Score 5 ★★★★★</option>
    <option value="4">Score 4 ★★★★</option>
    <option value="3">Score 3 ★★★</option>
    <option value="2">Score 2 ★★</option>
    <option value="1">Score 1 ★</option>
  </select>
  <span id="rowCount" style="color:var(--muted);font-size:.8rem"></span>
</div>
<div class="table-wrap">
  <table>
    <thead><tr>
      <th>#</th><th>Scenario</th><th>Case</th><th>🔒</th><th>Intent</th>
      <th>Satisfaction</th><th>Score</th><th>Mistakes</th><th>Detected</th><th>Reasoning</th>
    </tr></thead>
    <tbody id="dialogBody">{"".join(table_rows)}</tbody>
  </table>
</div>

</main>
<footer>Generated by report.py &nbsp;·&nbsp; Chart.js 4 &nbsp;·&nbsp; {total} dialogs analyzed</footer>

<script>
Chart.defaults.color='#94a3b8';
Chart.defaults.borderColor='#2e3352';
const SCORE_COLORS=['#ef4444','#f97316','#f59e0b','#84cc16','#22c55e'];
const SAT_COLORS  =['#22c55e','#f59e0b','#ef4444'];
const INTENT_PAL  =['#6366f1','#38bdf8','#a78bfa','#34d399','#fb923c','#f472b6'];
const MISTAKE_COLORS=['#ef4444','#f97316','#f59e0b','#a78bfa','#38bdf8'];

new Chart(document.getElementById('scoreChart'),{{
  type:'bar',
  data:{{labels:{_json.dumps([str(i) for i in range(1,6)])},
         datasets:[{{data:{_json.dumps([score_counts[i] for i in range(1,6)])},
           backgroundColor:SCORE_COLORS,borderRadius:6,borderSkipped:false}}]}},
  options:{{responsive:true,plugins:{{legend:{{display:false}},
    tooltip:{{callbacks:{{label:c=>` ${{c.raw}} dialogs`}}}}}},
    scales:{{y:{{beginAtZero:true,ticks:{{stepSize:1}}}},x:{{grid:{{display:false}}}}}}
  }}
}});

new Chart(document.getElementById('satChart'),{{
  type:'doughnut',
  data:{{labels:{_json.dumps(sat_labels)},
         datasets:[{{data:{_json.dumps(sat_values)},backgroundColor:SAT_COLORS,
           borderColor:'#1a1d27',borderWidth:3}}]}},
  options:{{responsive:true,cutout:'65%',
    plugins:{{legend:{{position:'bottom',labels:{{padding:16}}}},
      tooltip:{{callbacks:{{label:c=>` ${{c.label}}: ${{c.raw}}`}}}}}}
  }}
}});

new Chart(document.getElementById('intentChart'),{{
  type:'bar',
  data:{{labels:{_json.dumps(intent_labels)},
         datasets:[{{data:{_json.dumps(intent_values)},backgroundColor:INTENT_PAL,
           borderRadius:6,borderSkipped:false}}]}},
  options:{{indexAxis:'y',responsive:true,
    plugins:{{legend:{{display:false}}}},
    scales:{{x:{{beginAtZero:true,ticks:{{stepSize:1}}}},y:{{grid:{{display:false}}}}}}
  }}
}});
{mistakes_js_block}
{hda_js_block}

// Live table filtering
const rows=Array.from(document.querySelectorAll('#dialogBody tr'));
const search=document.getElementById('search');
const fSat=document.getElementById('filterSat');
const fScore=document.getElementById('filterScore');
const counter=document.getElementById('rowCount');
function applyFilters(){{
  const q=search.value.toLowerCase(),s=fSat.value,sc=fScore.value;
  let vis=0;
  rows.forEach(r=>{{
    const ok=(!q||r.textContent.toLowerCase().includes(q))
           &&(!s||r.dataset.sat===s)&&(!sc||r.dataset.score===sc);
    r.style.display=ok?'':'none';if(ok)vis++;
  }});
  counter.textContent=`${{vis}} / ${{rows.length}} dialogs`;
}}
search.addEventListener('input',applyFilters);
fSat.addEventListener('change',applyFilters);
fScore.addEventListener('change',applyFilters);
applyFilters();
</script>
</body>
</html>"""

def run_report(
    input_path: Path,
    out_path: Path,
    title: str,
    html_path: Path | None = None,
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
    print(f"✓ Markdown saved → {out_path}")

    effective_html = html_path if html_path else out_path.with_suffix(".html")
    html_content = build_html_report(data, title)
    effective_html.parent.mkdir(parents=True, exist_ok=True)
    with open(effective_html, "w", encoding="utf-8") as fh:
        fh.write(html_content)
    print(f"✓ HTML   saved → {effective_html}  (open in browser for interactive dashboard)")


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
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Override HTML output path.\n"
            "Default: same as --out but with .html extension.\n"
            "Example: results/report.html"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_report(
        input_path=args.input,
        out_path=args.out,
        title=args.title,
        html_path=args.html,
    )
