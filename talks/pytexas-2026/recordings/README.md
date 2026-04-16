# Asciinema fallback recordings

Recorded `.cast` files for each of the four talk beats, used as fallbacks when
the live demo fails (bad wifi, crashed process, timebox pressure).

## Files

| File                  | Beat                          | How it's recorded       |
| --------------------- | ----------------------------- | ----------------------- |
| `01_argparse.cast`    | argparse "before" script      | automated (see below)   |
| `02_typer_help.cast`  | Typer, no Rich                | automated               |
| `03_rich_cli.cast`    | Real `shapmonitor` CLI (Rich) | automated               |
| `04_watch_tui.cast`   | `shapmonitor watch` (Textual) | **live** — see below    |

## One-time setup

Demo data must exist at `demo/shap_logs/` with 14 days of partitioned SHAP
data. Regenerate if the directory is missing or stale:

```bash
poetry run python scripts/make_demo_data.py
```

All three automated record scripts `cd` into `talks/pytexas-2026/` and reach
the demo data through the `demo` symlink (which points at `../../demo`).
This keeps the recorded commands short and realistic — `python
01_before_argparse.py demo/batch_current.csv` instead of the full
`talks/pytexas-2026/...` path that would give away that this is talk staging.

## Recording the automated beats (1–3)

From the repo root:

```bash
asciinema rec --idle-time-limit 2 --overwrite \
    --command "bash talks/pytexas-2026/recordings/record_01_argparse.sh" \
    talks/pytexas-2026/recordings/01_argparse.cast

asciinema rec --idle-time-limit 2 --overwrite \
    --command "bash talks/pytexas-2026/recordings/record_02_typer_help.sh" \
    talks/pytexas-2026/recordings/02_typer_help.cast

asciinema rec --idle-time-limit 2 --overwrite \
    --command "bash talks/pytexas-2026/recordings/record_03_rich_cli.sh" \
    talks/pytexas-2026/recordings/03_rich_cli.cast
```

The `--command` form runs one command and exits — no interactive shell noise.
The `--idle-time-limit 2` caps `pause` delays at 2s on playback, so
deliberate pauses don't bore the audience.

## Recording beat 4 (the TUI) — live

The Textual watch app is an interactive TUI — we can't scripted-type
keystrokes into it meaningfully. Record it live. First `cd` into the talk
directory so the recorded `$ ` prompt and paths match beats 1–3:

```bash
cd talks/pytexas-2026
asciinema rec --idle-time-limit 2 --overwrite \
    recordings/04_watch_tui.cast
```

Once the recording shell opens, perform this exact sequence:

1. `clear`
2. `poetry run shapmonitor watch --data-dir demo/shap_logs`
3. Wait ~3s for the dashboard to settle.
4. Press `/` to open the filter input.
5. Type `age` — table filters to the `age` row only.
6. Press `Escape`, then `r` to manually refresh.
7. Press `q` to quit.
8. `exit` (or Ctrl-D) to stop the recording.

## Playback

Any `.cast` file plays in any terminal with asciinema installed:

```bash
asciinema play talks/pytexas-2026/recordings/03_rich_cli.cast
```

Speed up a too-slow recording with `--speed 2`. Pause with space.

## Re-recording a single beat

Because each recording is self-contained, re-recording one doesn't require
redoing the others. The `--overwrite` flag replaces the existing `.cast`.

## Tuning the typing speed

The scripts simulate typing at 20 characters per second (roughly fast-but-
human). Adjust with an env var:

```bash
TYPE_SPEED=30 asciinema rec --command "bash .../record_03_rich_cli.sh" ...
```
