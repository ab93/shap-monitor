---
marp: true
theme: default
paginate: true
backgroundColor: #e1e2e7
color: #1a1b26
style: |
  /* --- Base typography --- */
  section {
    font-family: '', 'SF Pro Text', -apple-system, system-ui, sans-serif;
    font-size: 26px;
    background-color: #e1e2e7;   /* Tokyo Night Day background */
    color: #1a1b26;              /* near-black foreground */
    padding: 28px 56px 40px;     /* trim top + bottom; keep side breathing room */
  }
  /* Kill the browser's default heading top-margin so h1 hugs the slide top */
  section > h1:first-child,
  section > h2:first-child { margin-top: 0; }
  section.lead {
    text-align: center;
    font-size: 32px;
  }
  section.lead h1 {
    font-size: 60px;
    color: #2e7de9;              /* Tokyo Night Day blue */
  }
  section.lead h2 {
    font-size: 32px;
    color: #4d5b94;
    font-weight: 300;
  }
  h1 {
    color: #2e7de9;
    font-size: 42px;
    border-bottom: 2px solid #2e7de955;
    padding-bottom: 8px;
  }
  h2 {
    color: #007197;              /* cyan */
    font-size: 30px;
  }
  h3 {
    color: #9854f1;              /* purple */
    font-size: 24px;
  }

  /* --- Inline code --- */
  code {
    font-family: 'JetBrains Mono', 'Fira Code', 'SF Mono', monospace;
    background: #d5d6db;
    color: #b15c00;              /* orange accent for inline */
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.92em;
  }

  /* --- Code blocks: balance readability with vertical space --- */
  pre {
    background: #f4f4f5 !important;   /* lighter than slide bg */
    border: 1px solid #c4c8da;
    border-radius: 8px;
    padding: 10px 14px !important;
    font-size: 20px;
    line-height: 1.35;
  }
  /* For slides with long code blocks — opt in via <!-- _class: dense --> */
  section.dense pre { font-size: 17px; line-height: 1.3; padding: 8px 12px !important; }
  pre code {
    background: transparent;
    color: #1a1b26;
    padding: 0;
    font-size: inherit;
  }

  /* --- Syntax highlighting (Tokyo Night Day palette) --- */
  .hljs-keyword,
  .hljs-built_in,
  .hljs-type { color: #9854f1; }                         /* purple: def, import, class, return */
  .hljs-string,
  .hljs-meta-string { color: #587539; }                  /* green: string literals */
  .hljs-number,
  .hljs-literal { color: #b15c00; }                      /* orange: numbers, True/False/None */
  .hljs-comment,
  .hljs-quote { color: #848cb5; font-style: italic; }    /* muted blue-gray: comments */
  .hljs-function .hljs-title,
  .hljs-title.function_,
  .hljs-title { color: #2e7de9; }                        /* blue: function names */
  .hljs-params { color: #8c6c3e; }                       /* yellow: parameter names */
  .hljs-variable,
  .hljs-attr,
  .hljs-property { color: #f52a65; }                     /* pink: attributes, self */
  .hljs-meta,
  .hljs-decorator { color: #007197; }                    /* cyan: @decorators */
  .hljs-symbol,
  .hljs-bullet,
  .hljs-section { color: #118c74; }
  .hljs-deletion { color: #f52a65; background: #f5d5db; }
  .hljs-addition { color: #587539; background: #d5edd5; }

  /* --- Links, emphasis, blockquote --- */
  a { color: #007197; }
  strong { color: #587539; }
  em { color: #9854f1; font-style: normal; }
  blockquote {
    border-left: 4px solid #2e7de9;
    padding-left: 16px;
    color: #4d5b94;
    font-style: italic;
  }

  /* --- Tables --- */
  table { font-size: 22px; }
  th { background: #d5d6db; color: #2e7de9; }
  td { background: #f4f4f5; }

  /* --- Section dividers --- */
  section.section-divider {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.section-divider h1 {
    font-size: 54px;
    border-bottom: none;
  }
  section.section-divider h2 {
    color: #4d5b94;
    font-weight: 300;
  }

  /* --- Helper classes --- */
  .pattern-box {
    background: #f4f4f5;
    border: 1px solid #2e7de944;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
---

<!-- _class: lead -->

# Building Professional CLIs in Python

**Avik Basu**
PyTexas 2026

<!-- speaker notes:
Welcome! Today we're going to take a CLI from basic print statements
to an interactive terminal dashboard — and along the way, pick up
patterns that make any CLI tool professional-grade.
-->

---

# About Me

- Based in Sunnyvale CA
- Staff Data Scientist at Intuit
- Editor at PyOpenSci
- Love RPG games
- Driving is therapy!

---

# About Me

- Based in Sunnyvale CA
- Staff Data Scientist at Intuit
- Editor at PyOpenSci
- Love RPG games
- Driving is therapy!

![bg right](mine.jpg)

---

# Agenda

1. **Why CLIs?** - Motivation
2. **Argparse** — Baseline functionality
3. **Typer** — function signatures as CLIs
4. **Rich** — output that communicates
5. **Textual** — interactive TUIs
6. **Production patterns** — how to scale things up!

<!-- speaker notes:
Four library stages in the order you'd naturally adopt them, plus the
cross-cutting production patterns slide set. The first three sections
motivate each other; the Textual section is optional if your tool doesn't
need interactivity. Total: ~25 minutes + Q&A.
-->

---

# My first CLI experience

<br>

- Year **2013**
- Just Installed **Ubuntu**
- First **Python project**
- Learning **Git** as a CLI

![bg right 100% contrast](git-status-2.png)

---

## I typed `git commit`...

...and suddenly I was **here**

<!-- speaker notes:
- A full-screen editor I had never seen
- No menus
- No "save" button
- No "exit" button
- Just a blinking cursor
-->

![bg right:70% 100% contrast](git-commit.png)

<!-- speaker notes:
This is the vi editor. Ubuntu's default. git opens it when you don't
pass -m for your commit message. I had no idea what I was looking at.
-->

---

## OK, I wrote a message...

...now **how do I save and exit?**

<!-- speaker notes:
- `Ctrl-S`? Nothing happened.
- `Ctrl-C`? It killed the whole thing — message lost.
- Esc? Just beeped at me.
-->

![bg right:70% 90% contrast](git-commit-2.png)

<!-- speaker notes:
I figured out insert mode eventually, typed "first commit wohoooo!",
then hit the wall again trying to actually save and exit.
-->

---

## This isn't vi's/vim's fault.

<br>

It's a beautifully designed tool.

The problem was **me** — a beginner dropped into a powerful tool with
no path from _"what is this?"_ to _"oh, I get it."_

**That's the gap great CLIs close: a gentle on-ramp for newcomers,
without compromising power for experts.**

<!-- speaker notes:
Quick acknowledgement for any vi enthusiasts in the room — this isn't
a dig. vi/vim is one of the most thoughtfully designed text editors
ever built. The issue wasn't vi's interface; it was that I, as a total
beginner, had no way to discover how it worked.

Pivot the lesson: good CLIs onboard beginners gracefully while still
respecting power users. The patterns we cover today are all about
making that first-run experience welcoming without dumbing things down.
-->

---

# Why great CLIs matter

<br>

- **Fast** — no browser, no GUI overhead
- **Composable** — pipe output, chain tools, automate
- **Scriptable** — run in CI, cron jobs, SSH sessions, servers, etc.
- **Universal** — works everywhere; can be made platform agnostic
- **Agent-ready** — AI agents are CLIs' newest users

<!-- speaker notes:
ML teams live in terminals — training, deploying, debugging.
If the terminal experience is bad, people build a web UI instead.
And now there's a new consumer: AI agents like Claude Code and
Gemini CLI invoke CLI tools, read --help, parse output.
The patterns we'll cover today serve both audiences.

"A good CLI is an API for humans — and increasingly, for machines."
-->

---

# Our Running Example

<br>

## A tool to **monitor ML model explainability**

- Capture explanations as the model runs in production
- Understand which features drive the predictions
- Compare explanations across different model versions or time windows

---

# Three Commands

<br>

| Command  | What it does                                      |
| -------- | ------------------------------------------------- |
| `log`    | Log explanation values for a batch of predictions |
| `report` | Generate summary and drift reports                |
| `watch`  | Live monitoring dashboard                         |

<br>

No ML background needed. Just think of it as:

**log data → analyze data → watch data live**

<!-- speaker notes:
The domain doesn't matter for what we're learning today.
These three commands naturally motivate all the patterns we'll cover.
-->

---

# What we want from our CLI

<br>

1. **Distributable in a venv and globally** — for library _and_ CLI users
2. **Support Subcommands** — `shapmonitor report summary`, `shapmonitor report drift`
3. **Type-safe inputs** — bad arguments rejected with helpful errors
4. **Config hierarchy** — explicit flags/args → env vars → defaults
5. **Progress feedback** — spinners, bars, "still working" signals
6. **Fast** — instant startup, never feels heavy
7. **Polished tables** — ranked, color-coded, easy to scan

<!-- speaker notes:
Seven criteria — the rubric for the rest of the talk. Walk through verbally:
which library helps us hit each criterion. Don't reveal the lazy-imports
trick yet — that's the payoff in the production patterns section.

Mapping for your reference:
- argparse: partial on (2), (3); nothing else
- Typer: (2), (3), (4) declaratively
- Rich: (5), (6)
- Packaging: (1) — pyproject.toml entry point + optional extras
- (7) Fast — library-agnostic, covered in production patterns
-->

---

<!-- _class: section-divider -->

# Argparse

Imperative Style

---

<!-- _class: dense -->

# The log command, with argparse

```python
# shapmonitor/__main__.py
import argparse, os, sys

def main():
    parser = argparse.ArgumentParser(
        prog="shapmonitor",
        description="Log SHAP values for a batch of predictions.",
    )
    parser.add_argument("data", help="Path to CSV file with feature data")
    parser.add_argument("--model", required=True, help="Path to model file")
    parser.add_argument("--data-dir", default="./shap_logs")
    # more args...

    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: data file not found: {args.data}", file=sys.stderr)
        sys.exit(1)
    # same for model
    if not 0.0 <= args.sample_rate <= 1.0:
        print("Error: --sample-rate must be between 0.0 and 1.0", file=sys.stderr)
        sys.exit(1)
    # ... and *now* we can read the CSV and compute SHAP values
```

**~20 lines of plumbing before we touch the data.**

---

## Help command - built-in

<br>

```bash
$ python -m shapmonitor log --help
```

<br>

```
usage: shapmonitor log [-h] --model MODEL [--data-dir DATA_DIR] [--sample-rate SAMPLE_RATE] [--model-version MODEL_VERSION] data

positional arguments:
  data                  Path to CSV file with feature data

options:
  -h, --help            show this help message and exit
  --model MODEL         Path to pickled model
  --data-dir DATA_DIR   Output directory
  --sample-rate SAMPLE_RATE
                        Sampling rate (0.0-1.0)
  --model-version MODEL_VERSION
                        Model version tag
```

---

# The `log` command

<br>

```bash
$ python -m shapmonitor log demo/batch_current.csv --model demo/model.pkl --data-dir ./logs
```

```
Logged 100 samples to ./logs
```

<br>

It writes the explanation (SHAP) values to disk.

---

# The `report summary` command

<br>

```
$ python -m shapmonitor report summary --data-dir ./logs

Feature               Mean |SHAP|       Mean        Std
------------------------------------------------------
age                        0.1159     0.0106     0.1298
recent_inquiries           0.1053    -0.0134     0.1165
employment_years           0.0806     0.0136     0.1001
credit_score               0.0563    -0.0075     0.0673
payment_history            0.0476    -0.0178     0.0612
debt_ratio                 0.0340    -0.0187     0.0427
income                     0.0120    -0.0003     0.0183
num_accounts               0.0054    -0.0001     0.0088
```

<br>

It works but, **we can do better**.

---

# The pain points

<br>

1. **Verbose argument definitions** — every flag is 2+ lines
2. **Manual validation** — you check types, ranges, file existence yourself
3. **No config hierarchy** — can't use environment variables without extra code
4. **Text-only output** — not super pleasing to the eye
5. **Raw tracebacks** — users don't need to see and debug

---

<!-- _class: section-divider -->

# Typer

<br>

## **Core Idea: The Function Signature _is_ the CLI**

---

```python
# shapmonitor/cli/log.py
from pathlib import Path
from typing import Annotated
import typer

def log_command(
    data: Annotated[Path, typer.Argument(exists=True, readable=True)],
    model: Annotated[Path, typer.Option("--model", "-m",
        exists=True, readable=True)],
    data_dir: Annotated[Path, typer.Option(
        envvar="SHAPMONITOR_DATA_DIR")] = Path("./shap_logs"),
    sample_rate: Annotated[float, typer.Option(min=0.0, max=1.0)] = 1.0,
) -> None:
    """Log SHAP values for a batch of predictions."""
    ...
```

<br>

**Type hints** become **CLI arguments**. Validation is **Declarative.**

---

# Side by side

<div class="columns">
<div>

### argparse

```python
parser.add_argument(
    "data",
    type=str,
    help="Path to CSV"
)
# Then manually:
# - check file exists
# - convert to Path
# - validate readable
```

</div>
<div>

### Typer

```python
data: Annotated[
    Path,
    typer.Argument(
        help="Input CSV file.",
        exists=True,
        readable=True,
    )
]
# Done. ✓
```

</div>
</div>

**Same thing, but Typer validates the path exists, is readable, and gives a helpful error if not.**

---

# Subcommands and composition

```python
# shapmonitor/cli/report.py — a nested Typer group
report_app = typer.Typer(no_args_is_help=True)

@report_app.command("summary")
def summary_command(...): ...

@report_app.command("drift")
def drift_command(...): ...

# shapmonitor/cli/__init__.py — mount the group under "report"
app.add_typer(report_app, name="report")
```

<br>

```bash
-> shapmonitor report summary --top-k 5
-> shapmonitor report drift --ref last-14d..last-7d
```

**Nested subcommands with zero boilerplate.**

---

# Sensible Typer defaults

<br>

Users run `--help` constantly. It should be instant and friendly.

```python
# shapmonitor/cli/__init__.py
app = typer.Typer(
    name="shapmonitor",
    no_args_is_help=True,     # no args? show help, don't error
    help="Monitor SHAP explanations over time.",
    rich_markup_mode="rich",  # styled help output
)
```

- **`no_args_is_help=True`** — bare `shapmonitor` prints help, not a usage error
- **`rich_markup_mode="rich"`** — colored, boxed help output from the docstring

---

![bg fit](shapmonitor-help.png)

---

<!-- _class: section-divider -->

# Enhanced Output

## Rich

---

# More than just a better UX

<br>

Rich gives you:

- **Progress bars** — feedback for long-running tasks
- **Tables** — structured data that's easy to scan
- **Panels** — grouped information with borders
- **Styled text** — emphasis, color-coded severity
- **Tracebacks** — readable, syntax-highlighted errors

---

# Progress bars

```python
# shapmonitor/cli/log.py
from rich.progress import Progress, SpinnerColumn, TextColumn

with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    console=console,
    transient=True,  # disappears when done
) as progress:
    progress.add_task("Computing SHAP values...", total=None)
    monitor.log_batch(X)
```

```
⠋ Computing SHAP values...
```

**Transient spinners for indeterminate work. Progress bars when you know the total.**
Users know something is happening — that's the entire point.

---

# Tables with visual weight

```python
# shapmonitor/cli/report.py — summary_command
from rich.table import Table

table = Table(title=f"SHAP Summary  ({p.start} → {p.end})")
table.add_column("Feature", style="bold")
table.add_column("Mean |SHAP|", justify="right")
table.add_column("", justify="left")       # inline bar
table.add_column("Mean", justify="right")
table.add_column("Std", justify="right")

for feat, row in summary_df.iterrows():
    bar_len = int(20 * row["mean_abs"] / max_mean_abs)
    bar = "[cyan]" + "\u2588" * bar_len + "[/cyan]"
    table.add_row(str(feat), f"{row['mean_abs']:.4f}", bar,
                  f"{row['mean']:.4f}", f"{row['std']:.4f}")

console.print(table)
```

---

![bg fit](report-summary.png)

---

# Styled status and panels

<br>

```python
# shapmonitor/cli/report.py — drift_command
from rich.panel import Panel

alerts = int((drift_df["psi"] >= 0.25).sum())
warnings = int(((drift_df["psi"] >= 0.1) & (drift_df["psi"] < 0.25)).sum())
stable = int((drift_df["psi"] < 0.1).sum())

headline = " · ".join([
    f"[red]{alerts} alert{'s' if alerts != 1 else ''}[/red]",
    f"[yellow]{warnings} warning{'s' if warnings != 1 else ''}[/yellow]",
    f"[green]{stable} stable[/green]",
])

console.print(Panel(headline, title="Drift Report", border_style="blue"))
```

---

![bg fit](report-drift.png)

---

<!-- _class: section-divider -->

# Interactive TUI

## Textual

---

# When static output isn't enough

<br>

Some tasks need **interactivity**:

- **Live monitoring** — data that updates in real time
- **Exploration** — filter, sort, navigate large datasets
- **Keyboard-driven workflows** — no mouse, no re-running

**Textual** gives you a full application framework in the terminal:
widgets, layouts, events, CSS styling, keybindings.

---

# App structure

```python
# shapmonitor/cli/_watch_app.py
from textual.app import App, ComposeResult
from textual.widgets import DataTable, Footer, Header, Input

class WatchApp(App):
    TITLE = "shapmonitor watch"

    CSS = """
    #filter-input { dock: top; margin: 0 1; height: 3; }
    DataTable { height: 1fr; }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield Input(placeholder="Type to filter features...",
                    id="filter-input")
        yield DataTable(id="shap-table")
        yield Footer()
```

**`compose()` defines the layout. CSS controls sizing.
That's your entire UI skeleton.**

---

# Live data with auto-refresh

```python
def on_mount(self) -> None:
    table = self.query_one(DataTable)
    table.add_columns("Feature", "Mean |SHAP|", "Std", "PSI", "Status")

    self._load_data()                                  # initial load
    self.set_interval(self._refresh_interval, self._load_data)

def _load_data(self) -> None:
    analyzer = SHAPAnalyzer(get_backend(self._data_dir))
    period = parse_period(self._period_spec)
    self._summary = analyzer.summary(period.start, period.end)
    self._refresh_table()
```

**`set_interval()` — that's all it takes for live updates.**
The table rebuilds with fresh data every 5 seconds.

---

# Interactive filtering

```python
from textual.binding import Binding

BINDINGS = [
    Binding("q", "quit", "Quit"),
    Binding("r", "refresh", "Refresh"),
    Binding("/", "focus_filter", "Filter"),
    Binding("escape", "clear_filter", "Clear filter"),
]

def on_input_changed(self, event: Input.Changed) -> None:
    self._filter_text = event.value.lower()
    self._refresh_table()

def _refresh_table(self) -> None:
    table = self.query_one(DataTable)
    table.clear()
    for feature, row in self._summary.iterrows():
        if self._filter_text not in str(feature).lower():
            continue
        table.add_row(str(feature), ...)
```

**Type to filter. Press `/` to focus. `Escape` to clear. `q` to quit.**

---

# Pattern: progressive disclosure

**Simple by default, flexible when needed.**

```bash
# Basic — just works
$ shapmonitor report summary

# More control
$ shapmonitor report summary --top-k 5 --period last-30d

# Full control
$ shapmonitor report drift \
    --ref last-14d..last-7d \
    --curr last-7d..now \
    --top-k 10 --json | jq .
```

In TUIs, same principle:

- App **starts showing everything**
- User **filters down** with keyboard
- Power features via **keybindings**, not required

---

<!-- _class: section-divider -->

# Production Patterns

## Cross-cutting hygiene that makes any CLI feel professional

---

# Composability — pipeable output

A great CLI plays well with `jq`, `xargs`, **and** humans at the same time.

```python
if json_mode:
    typer.echo(json.dumps(data, default=str))
    return
render_rich_table(data)
```

```bash
$ shapmonitor report summary --json | jq '.features[:3]'
```

```bash
[
  {
    "feature": "recent_inquiries",
    "mean_abs": 0.11099565029144287,
    "mean": 0.02865222841501236,
    "std": 0.11797577887773514,
    "min": -0.2615908980369568,
    "max": 0.23528572916984558
  },
]
```

---

# Imports are side effects

Every import line is **executable code** — it loads modules, runs their top-level statements, and triggers the entire import chain.

```python
# shapmonitor/__init__.py — the obvious version
from shapmonitor.monitor import SHAPMonitor
# ↑ pulls in shap → numba → llvmlite → ~80 transitive deps
```

The cost: every `import shapmonitor` pays for SHAP — even when the caller never touches `SHAPMonitor`.

For a CLI, that means **`shapmonitor --help` loads numba just to print docs.**

<!-- speaker notes:
The mental model: imports aren't declarative ("foo is now available"); they're imperative ("execute foo's module body"). Each import is a side effect — it runs code, allocates memory, registers handlers.

For libraries that depend on heavy ML stacks (shap, numba, sklearn), this compounds. shapmonitor's eager version takes 430ms just to import — before any function runs. CLIs are the worst victims of this because --help should be instant.

Set up the next slide: "There's a Python feature that lets the library defer this — PEP 562."
-->

---

# PEP 562 — lazy module attributes

```python
# shapmonitor/__init__.py
__all__ = ["SHAPMonitor"]

def __getattr__(name):
    if name == "SHAPMonitor":
        from shapmonitor.monitor import SHAPMonitor
        return SHAPMonitor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- `import shapmonitor` → loads only the module shell
- `shapmonitor.SHAPMonitor` → triggers `shap → numba → llvmlite` **only on first access**

Same public API. `from shapmonitor import SHAPMonitor` still works.
**The library decides when to pay the import cost — not the importer.**

<!-- speaker notes:
PEP 562 was added in Python 3.7 (2018) but most developers don't know it exists. It lets you define a module-level __getattr__ that intercepts attribute access on the module itself.

Two ways to think about it: (1) lazy property for modules, (2) the import system's equivalent of __getattr__ on a class.

The trick is the `from shapmonitor.monitor import SHAPMonitor` happens INSIDE __getattr__, so it only fires the first time someone accesses shapmonitor.SHAPMonitor. After that, Python caches the attribute on the module — so subsequent accesses are free.

Add it to any package's __init__.py. Costs zero for callers that don't access the heavy attributes (e.g., --help). Costs the same as before for callers that do.
-->

---

<!-- _class: dense -->

# Optional CLI dependencies

The underlying library can have optional CLI dependencies

```toml
# pyproject.toml
[project.scripts]
shapmonitor = "shapmonitor.cli:app"   # ← puts `shapmonitor` on $PATH

[project.optional-dependencies]
cli = [
    "typer>=0.12,<1.0",
    "rich>=13.0,<15.0",
    "textual>=1.0,<9.0; python_version < '4'",
]
```

```python
# shapmonitor/cli/__init__.py
try:
    import typer
except ImportError:
    sys.stderr.write("Install with: pip install shap-monitor[cli]\n")
    sys.exit(1)
```

**`pip install shap-monitor` → just the library**
**`pip install shap-monitor[cli]` → library + `shapmonitor` command on `$PATH`**

---

# Config hierarchy

**Flags → env vars → defaults** — the precedence users expect.

```python
data_dir: Annotated[
    Path,
    typer.Option(
        envvar="SHAPMONITOR_DATA_DIR",   # ← falls back to env var
    ),
] = Path("./shap_logs")                 # ← falls back to default
```

```bash
# All three work, in priority order:
$ shapmonitor log data.csv --data-dir /custom/path
$ SHAPMONITOR_DATA_DIR=/custom/path shapmonitor log data.csv
$ shapmonitor log data.csv  # uses ./shap_logs
```

**One line gives you three configuration layers.**

---

# When to reach for what

<br>

| You need...                 | Use...        | Complexity |
| --------------------------- | ------------- | ---------- |
| Just argument parsing       | **Typer**     | Low        |
| Readable, structured output | + **Rich**    | Low        |
| Progress bars, spinners     | + **Rich**    | Low        |
| Live data, interactivity    | + **Textual** | Medium     |
| Full TUI application        | + **Textual** | Higher     |

<br>

**You don't have to use all three.** Start with Typer, add Rich when output matters, reach for Textual when you need interactivity.

Each layer is independent — adopt incrementally.

---

<!-- _class: section-divider -->

<!-- _class: lead -->

# Thank you!

**Libraries**: typer.tiangolo.com · rich.readthedocs.io · textual.textualize.io

**Live, shipping example**: `pip install shap-monitor[cli]` · github.com/ab93/shap-monitor

**These slides**: github.com/ab93/shap-monitor/tree/main/talks/pytexas-2026

<!-- speaker notes:
Mention shap-monitor briefly as "a project where I use all three
of these libraries together — check it out if you want a real-world example."
-->
