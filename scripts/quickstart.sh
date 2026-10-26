#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e ".[dev]"
pytest -q
ldlab train --config configs/smoke.yaml
