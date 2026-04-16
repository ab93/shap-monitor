# PyTexas 2026 — From Scripts to TUIs

Talk materials for *"From Scripts to TUIs: Building Better Python CLIs with Typer, Rich, and Textual"* (PyTexas 2026).

## The story

The talk walks the audience from an afternoon-hack argparse script to the production `shapmonitor` CLI. Every step is a real, runnable file so attendees can `git log` the progression rather than squinting at slides.

| Step | File                       | What it shows                                                      |
| ---- | -------------------------- | ------------------------------------------------------------------ |
| 1    | `01_before_argparse.py`    | The "before": argparse + manual validation + hand-padded output    |
| 2    | `02_typer_no_rich.py`      | Argparse → Typer. Output stays plain. Isolates the Typer win.      |
| 3    | `shapmonitor/cli/` (Rich)  | Typer subcommand split + Rich `Table`/`Panel`/spinner              |
| 4    | `shapmonitor/cli/_watch_app.py` | Textual `WatchApp` — live dashboard for streaming SHAP data   |

Each step's diff against the previous is the talk's teaching substrate. Keeping the diffs narrow — one library at a time — lets attendees see exactly what each library buys you.

### Step 1 → Step 2 (argparse → Typer)

What vanished:
- `argparse.ArgumentParser(...)` + `parser.add_argument(...)` boilerplate
- Manual `os.path.exists(...)` checks → `exists=True` on the `Path` param
- Manual `float(args.sample_rate)` coercion → `float` type hint auto-coerces
- Blanket `try/except Exception` → Typer owns exit codes + tracebacks
- The `sys` import entirely

What came for free:
- Colored, boxed `--help` output generated from the docstring + param `help=` strings
- POSIX-correct exit codes: `2` for misuse, `1` for runtime failure (vs argparse's flat `1`)
- Validation error messages that name the offending parameter (`Invalid value for 'DATA': ...`)

### Step 2 → Step 3 (Typer → Typer + Rich + subcommand split)

Two things happen at once (deliberately — they motivate each other):
- Output gets colored, ranked, and time-aware via `Rich.Table` + `Panel`
- The monolith splits: `log` and `report summary` become separate subcommands with single responsibilities. The summary table gets cheaper to extend — extra columns (`Mean`, `Std`) that the f-string version never bothered with appear naturally.

### Step 3 → Step 4 (polling reports → live TUI)

`shapmonitor watch` is a Textual `App` that polls the parquet backend every 5 seconds and renders a filterable `DataTable`. Showcases:
- Textual's declarative widget composition
- `set_interval` for timer-driven refresh
- Keyboard bindings (`q`, `r`, `/`) — the kind of affordance a CLI can't offer

## Running each step

```bash
# Once — installs shap-monitor + CLI extras into the poetry venv
poetry install --extras cli

# Step 1 — argparse
bash talks/pytexas-2026/sample_command.sh

# Step 2 — Typer, no Rich (same shape, run directly)
poetry run python talks/pytexas-2026/02_typer_no_rich.py demo/batch_current.csv \
    --model demo/model.pkl \
    --data-dir /tmp/typer_demo_logs

# Step 3 — the real CLI (Typer + Rich + subcommand split)
poetry run shapmonitor log demo/batch_current.csv \
    --model demo/model.pkl \
    --data-dir /tmp/shapmonitor_demo_logs
poetry run shapmonitor report summary \
    --data-dir /tmp/shapmonitor_demo_logs --period last-7d

# Step 4 — live dashboard
poetry run shapmonitor watch --data-dir /tmp/shapmonitor_demo_logs
```

Each step writes to its own `/tmp/*_demo_logs` directory so you can run them in any order during rehearsal without state bleeding between demos.
