#!/usr/bin/env bash
# Beat 3 of the talk arc — the real shapmonitor CLI. Typer + Rich + subcommand split.
#
# What the audience sees:
#   1. ``shapmonitor --help`` — the subcommand tree (``log``, ``report``, ``watch``).
#   2. ``shapmonitor log ...`` — Rich progress spinner + success panel.
#   3. ``shapmonitor report summary`` — Rich Table with inline bars, ranked.
#   4. ``shapmonitor report drift`` — Rich Table with color-coded PSI,
#      demonstrating that real drift exists in days 8–14 (injected by
#      ``scripts/make_demo_data.py``).
#
# Record with:
#   asciinema rec --idle-time-limit 2 --overwrite \
#       --command "bash talks/pytexas-2026/recordings/record_03_rich_cli.sh" \
#       talks/pytexas-2026/recordings/03_rich_cli.cast
#
# Note: ``set -u`` only (no ``-e``) to match beats 1–2 and survive any
# non-zero exits we want on screen.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TALK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TALK_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_demo_magic.sh"

clear

pe "poetry run shapmonitor --help"
pause 4

pe "poetry run shapmonitor log \\
    demo/batch_current.csv \\
    --model demo/model.pkl \\
    --data-dir demo/shap_logs"
pause 3

pe "poetry run shapmonitor report summary \\
    --data-dir demo/shap_logs \\
    --period last-7d"
pause 4

pe "poetry run shapmonitor report drift \\
    --data-dir demo/shap_logs \\
    --ref last-14d..last-7d \\
    --curr last-7d..now"
pause 4
