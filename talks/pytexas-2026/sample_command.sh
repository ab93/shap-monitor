#!/usr/bin/env bash
# The invocation shown on the "before" slide. Run from the repo root.
set -euo pipefail

python -m shapmonitor \
    demo/batch_current.csv \
    --model demo/model.pkl \
    --data-dir /tmp/argparse_demo_logs
