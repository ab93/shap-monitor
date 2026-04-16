#!/usr/bin/env bash
# Beat 1 of the talk arc — the argparse "before" script.
#
# What the audience sees:
#   1. A healthy run of the argparse version, producing a hand-padded table.
#   2. A broken invocation (``--sample-rate abc``), demonstrating argparse's
#      flat ``exit 1`` for both misuse and runtime errors.
#
# Record with:
#   asciinema rec --idle-time-limit 2 --overwrite \
#       --command "bash talks/pytexas-2026/recordings/record_01_argparse.sh" \
#       talks/pytexas-2026/recordings/01_argparse.cast
#
# Note: ``set -u`` (undefined-var catch) only — no ``-e``, because the error-
# path demo below intentionally exits non-zero and we want the script to
# continue so ``echo $?`` shows the exit code.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run from the talks directory so the recorded commands look like what a real
# user would type — ``python 01_before_argparse.py demo/...`` instead of
# ``python talks/pytexas-2026/01_before_argparse.py ...``. The ``demo``
# symlink inside talks/pytexas-2026/ points at the repo-level ``demo/``.
TALK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${TALK_DIR}"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/_demo_magic.sh"

clear

pe "python 01_before_argparse.py \\
    demo/batch_current.csv \\
    --model demo/model.pkl \\
    --data-dir /tmp/argparse_demo_logs"
pause 3

# Error-path: non-numeric --sample-rate trips our hand-rolled float() coercion.
pe "python 01_before_argparse.py \\
    demo/batch_current.csv \\
    --model demo/model.pkl \\
    --sample-rate abc"
pause 1

pe "echo \"Exit code: \$?\""
pause 3
