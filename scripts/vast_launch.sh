#!/usr/bin/env bash
# Rent the cheapest matching Vast.ai GPU, run the GPU smoke test + AR baseline, then auto-destroy.
#   scripts/vast_launch.sh launch           # needs the pushed REF on REPO_URL, and `vastai` + python3
#   scripts/vast_launch.sh collect <id>     # copy results/ + runs/ (checkpoints) to results/vast/<id>/, then destroy
# Budget: only offers <= MAX_DPH $/h are considered, and a watchdog on the box stops it after
# MAX_HOURS no matter what, so the worst case is MAX_DPH * MAX_HOURS (printed before renting)
# plus a little disk storage until you run `collect`. The watchdog stops (not destroys) so nothing is lost.
set -euo pipefail
GPU="${GPU:-RTX_4090}"; MAX_DPH="${MAX_DPH:-0.60}"; MAX_HOURS="${MAX_HOURS:-2}"; DISK="${DISK:-40}"
REPO_URL="${REPO_URL:-https://github.com/DarthVendor/latent-diffusion-lab}"
REF="${REF:-$(git rev-parse --abbrev-ref HEAD)}"
# default job = Qwen AR benchmark; JOB=experiment runs scripts/run_experiment.sh (env TARGET/STEPS/SEEDS/EVAL_N pass through)
if [ "${JOB:-bench}" = experiment ]; then
  JOB_CMD="TARGET=${TARGET:-answer} STEPS=${STEPS:-3000} SEEDS='${SEEDS:-0 1 2}' EVAL_N=${EVAL_N:-500} bash scripts/run_experiment.sh"
elif [ "${JOB:-bench}" = probe ]; then
  JOB_CMD="python scripts/draft_probe.py
touch results/DONE"
else
  JOB_CMD="python scripts/bench_qwen_ar.py --smoke --out results/ar_smoke.json
python scripts/bench_qwen_ar.py --out results/ar_baseline.json
python scripts/bench_qwen_ar.py --compiled --smoke --out results/ar_compiled_smoke.json
python scripts/bench_qwen_ar.py --compiled --out results/ar_compiled.json
touch results/DONE"
fi
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
# watchdog first, so a hung job can never outlive the budget; it STOPS (keeps the disk) rather than destroys, so checkpoints survive for collect
nohup bash -c 'sleep $((MAX_HOURS * 3600)); vastai set api-key \$CONTAINER_API_KEY; vastai stop instance \$CONTAINER_ID' >/dev/null 2>&1 &
set -ex
nvidia-smi
git clone --depth 1 --branch $REF $REPO_URL $R && cd $R
apt-get update -qq && apt-get install -y -qq build-essential   # torch.compile needs a C compiler; the runtime image has none
pip install -r requirements-gpu.txt && pip install -e .
mkdir -p results runs
$JOB_CMD
ONSTART
  vastai create instance "$id" --image "$IMAGE" --disk "$DISK" --onstart "$onstart" --raw
  echo "then: vastai logs <instance id>;  scripts/vast_launch.sh collect <instance id>"
  ;;
collect)
  [ -n "${2:-}" ] || { echo "usage: $0 collect <instance id>" >&2; exit 1; }
  mkdir -p "results/vast/$2"
  # `vastai copy` (rsync daemon) failed for us; plain scp over the instance's ssh works once our key is attached
  vastai attach ssh "$2" "$(cat ~/.ssh/id_ed25519.pub)" >/dev/null || true
  u=$(vastai ssh-url "$2"); h=${u#ssh://root@}; host=${h%:*}; port=${h##*:}
  # results/ holds JSON, runs/ holds training checkpoints; destroy only after both copies succeed
  for d in results runs; do
    for i in 1 2 3 4 5; do   # a freshly attached key takes a few seconds to apply
      scp -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=no -P "$port" -r \
        "root@$host:$R/$d/" "results/vast/$2/" 2>/dev/null && continue 2 || sleep 10
    done
    echo "copy of $d/ failed; instance kept" >&2; exit 1
  done
  [ -e "results/vast/$2/results/DONE" ] || { echo "no DONE marker: job unfinished or failed; instance kept (watchdog still armed)" >&2; exit 1; }
  echo y | vastai destroy instance "$2"   # destroy prompts for confirmation
  ;;
*) echo "usage: $0 launch | collect <id>" >&2; exit 1 ;;
esac
