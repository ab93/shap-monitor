#!/usr/bin/env bash
# The invocation shown on the "before" slide. Run from the repo root.
set -euo pipefail

python talks/pytexas-2026/01_before_argparse.py \
    demo/batch_current.csv \
    --model demo/model.pkl \
    --data-dir /tmp/argparse_demo_logs \
    --sample-rate 1.0 \
    --model-version v1-argparse
