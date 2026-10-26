#!/usr/bin/env bash
# Rent the cheapest matching Vast.ai GPU, run the GPU smoke test + AR baseline, then auto-destroy.
#   scripts/vast_launch.sh launch           # needs the pushed REF on REPO_URL, and `vastai` + python3
#   scripts/vast_launch.sh collect <id>     # copy results to results/vast/<id>/ and destroy the instance
# Budget: only offers <= MAX_DPH $/h are considered, and a watchdog on the box destroys it after
# MAX_HOURS no matter what, so the worst case is MAX_DPH * MAX_HOURS (printed before renting).
set -euo pipefail
GPU="${GPU:-RTX_4090}"; MAX_DPH="${MAX_DPH:-0.60}"; MAX_HOURS="${MAX_HOURS:-2}"; DISK="${DISK:-40}"
REPO_URL="${REPO_URL:-https://github.com/DarthVendor/latent-diffusion-lab}"
REF="${REF:-$(git rev-parse --abbrev-ref HEAD)}"
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
R=/workspace/latent-diffusion-lab

case "${1:-}" in
launch)
  git ls-remote --exit-code "$REPO_URL" "$REF" >/dev/null || { echo "push '$REF' to $REPO_URL first" >&2; exit 1; }
  id=$(vastai search offers "gpu_name=$GPU num_gpus=1 reliability>0.98 dph<=$MAX_DPH disk_space>=$DISK inet_down>=200 rentable=true" -o dph --raw |
       python3 -c 'import json,sys; o=json.load(sys.stdin); print(o[0]["id"] if o else "")')
  [ -n "$id" ] || { echo "no $GPU offer under \$$MAX_DPH/h" >&2; exit 1; }
  echo "cap: \$$MAX_DPH/h x ${MAX_HOURS}h = \$$(python3 -c "print(round($MAX_DPH*$MAX_HOURS,2))") max (offer $id)"
  onstart=$(mktemp)
  cat > "$onstart" <<ONSTART
#!/bin/bash
# watchdog first, so a hung job can never outlive the budget
nohup bash -c 'sleep $((MAX_HOURS * 3600)); vastai set api-key \$CONTAINER_API_KEY; vastai destroy instance \$CONTAINER_ID' >/dev/null 2>&1 &
set -ex
nvidia-smi
git clone --depth 1 --branch $REF $REPO_URL $R && cd $R
pip install -r requirements-gpu.txt && pip install -e .
mkdir -p results
python scripts/bench_qwen_ar.py --smoke --out results/ar_smoke.json    # fail fast if the GPU path is broken
python scripts/bench_qwen_ar.py --out results/ar_baseline.json
touch results/DONE
ONSTART
  vastai create instance "$id" --image "$IMAGE" --disk "$DISK" --onstart "$onstart" --raw
  echo "then: vastai logs <instance id>;  scripts/vast_launch.sh collect <instance id>"
  ;;
collect)
  [ -n "${2:-}" ] || { echo "usage: $0 collect <instance id>" >&2; exit 1; }
  mkdir -p "results/vast/$2"
  vastai copy "C.$2:$R/results/" "results/vast/$2/"
  [ -e "results/vast/$2/DONE" ] || { echo "no DONE marker: job unfinished or failed; instance kept (watchdog still armed)" >&2; exit 1; }
  vastai destroy instance "$2"
  ;;
*) echo "usage: $0 launch | collect <id>" >&2; exit 1 ;;
esac
