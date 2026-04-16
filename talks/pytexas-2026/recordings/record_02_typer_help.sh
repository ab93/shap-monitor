#!/usr/bin/env bash
# Beat 2 of the talk arc — Typer replaces argparse. No Rich yet.
#
# What the audience sees:
#   1. ``--help`` output — colored, boxed, from docstring + param help strings.
#   2. A clean run showing the output is the same hand-padded f-string as
#      step 1 (so step 3's Rich diff is uncontaminated).
#   3. The same ``--sample-rate abc`` error from beat 1 — but now Typer
#      names the offending parameter and exits with POSIX 2 (misuse), not 1.
#
# Record with:
#   asciinema rec --idle-time-limit 2 --overwrite \
#       --command "bash talks/pytexas-2026/recordings/record_02_typer_help.sh" \
#       talks/pytexas-2026/recordings/02_typer_help.cast
#
# Note: ``set -u`` only — no ``-e`` (error-path demo intentionally exits 2).
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TALK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TALK_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_demo_magic.sh"

clear

pe "poetry run python 02_typer_no_rich.py --help"
pause 4

pe "poetry run python 02_typer_no_rich.py \\
    demo/batch_current.csv \\
    --model demo/model.pkl \\
    --data-dir /tmp/typer_demo_logs"
pause 3

# Same bad --sample-rate as beat 1. Compare: Typer names the param and exits 2.
pe "poetry run python 02_typer_no_rich.py \\
    demo/batch_current.csv \\
    --model demo/model.pkl \\
    --sample-rate abc"
pause 1

pe "echo \"Exit code: \$?\""
pause 3
