#!/usr/bin/env bash
set -euo pipefail
export $(grep -v '^#' .env | xargs)
python main.py
