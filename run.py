"""
run.py
------
Single-command pipeline:
  1. generate.py  – generate synthetic support chats
  2. analyze.py   – analyze each dialog with an LLM
  3. report.py    – build an interactive HTML dashboard
  4. HTTP server  – serve the dashboard at http://localhost:<port>

Usage:
    python run.py
    python run.py --provider groq
    python run.py --provider gemini --count 25 --port 8080
    python run.py --skip-generate   # re-analyze existing chats.json
"""

import argparse
import http.server
import os
import signal
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
HTML_REPORT = RESULTS_DIR / "report.html"

def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"

def bold(t: str) -> str:  return _c("1", t)
def green(t: str) -> str: return _c("32", t)
def cyan(t: str) -> str:  return _c("36", t)
def red(t: str) -> str:   return _c("31", t)
def dim(t: str) -> str:   return _c("2", t)

def step(title: str) -> None:
    width = 60
    print()
    print(bold(cyan("─" * width)))
    print(bold(cyan(f"  {title}")))
    print(bold(cyan("─" * width)))


def run_step(cmd: list[str], label: str) -> None:
    """Run a sub-command, streaming its output. Exits on failure."""
    step(label)
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(red(f"\nStep failed: {label}"), file=sys.stderr)
        sys.exit(result.returncode)
    print(green(f"\nDone: {label}"))



class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Serve from results/ and suppress access logs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(RESULTS_DIR), **kwargs)

    def log_message(self, format: str, *args) -> None:  
        pass  
    def do_GET(self) -> None:
        if self.path in ("/", ""):
            self.send_response(302)
            self.send_header("Location", "/report.html")
            self.end_headers()
            return
        super().do_GET()


def serve(port: int, open_browser: bool) -> None:
    """Start HTTP server in a background thread, then block until Ctrl-C."""
    server = http.server.HTTPServer(("localhost", port), _QuietHandler)
    url = f"http://localhost:{port}/report.html"

    step(f"Starting dashboard at {url}")
    print(f"  Press {bold('Ctrl-C')} to stop.\n")

    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    print(green(f"  Dashboard → {bold(url)}"))
    print(dim("  (serving from results/)"))

    try:
        signal.pause()          
    except (KeyboardInterrupt, AttributeError):
        pass                    
    finally:
        print(dim("\n  Shutting down …"))
        server.shutdown()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full pipeline: generate → analyze → report → serve dashboard",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--provider",
        metavar="NAME",
        default=None,
        help=(
            "LLM provider to use for both steps.\n"
            "Options: groq · gemini · deepseek · mistral · together · openrouter · openai\n"
            "Default: auto-detected from .env"
        ),
    )
    p.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="Override the default model for the chosen provider.",
    )
    p.add_argument(
        "--count",
        type=int,
        default=25,
        metavar="N",
        help="Number of dialogs to generate (default: 25).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        metavar="PORT",
        help="Port for the local dashboard server (default: 8000).",
    )
    p.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip the generate step (use existing data/chats.json).",
    )
    p.add_argument(
        "--skip-analyze",
        action="store_true",
        help="Skip the analyze step (use existing results/analysis.json).",
    )
    p.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open the browser.",
    )
    p.add_argument(
        "--title",
        default="Support Chat Quality Report",
        help='Dashboard title (default: "Support Chat Quality Report").',
    )
    return p.parse_args()


def build_cmd(script: str, args: argparse.Namespace, extra: list[str] | None = None) -> list[str]:
    cmd = [sys.executable, script]
    if args.provider:
        cmd += ["--provider", args.provider]
    if args.model:
        cmd += ["--model", args.model]
    if extra:
        cmd += extra
    return cmd


def main() -> None:
    args = parse_args()

    print(bold("\nSupport Chat Quality Analyzer – run"))

    provider_label = args.provider or "auto-detect"
    print(f"\n  Provider : {cyan(provider_label)}")
    print(f"  Count    : {args.count} dialogs")
    print(f"  Port     : {args.port}")

    # Step 1 – generate
    if not args.skip_generate:
        run_step(
            build_cmd("generate.py", args, ["--count", str(args.count), "--resume"]),
            "Step 1/3 – Generating dialogs",
        )
    else:
        print(dim("\n  [skip] generate step"))

    # Step 2 – analyze
    if not args.skip_analyze:
        run_step(
            build_cmd("analyze.py", args, ["--resume"]),
            "Step 2/3 – Analyzing dialogs",
        )
    else:
        print(dim("\n  [skip] analyze step"))

    # Step 3 – build HTML report (no markdown)
    run_step(
        [
            sys.executable, "report.py",
            "--html", str(HTML_REPORT),
            "--out", str(RESULTS_DIR / "_report_unused.md"),
            "--title", args.title,
        ],
        "Step 3/3 – Building HTML dashboard",
    )

    # remove the unused markdown file
    unused_md = RESULTS_DIR / "_report_unused.md"
    if unused_md.exists():
        unused_md.unlink()

    # Step 4 – serve
    serve(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
