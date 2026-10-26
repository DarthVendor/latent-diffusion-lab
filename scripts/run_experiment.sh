#!/usr/bin/env bash
# On-box job: plain-Qwen baselines once, then for each variant (4 latent steps, then 0-latent ablation)
# run SEEDS training seeds (one at a time unless CONC>1 and the GPU is big enough). Env: TARGET=answer|solution STEPS=3000 SEEDS="0 1 2" EVAL_N=500 SKIP_AR=1 (skip the baselines)
set -ex
TARGET="${TARGET:-answer}"; STEPS="${STEPS:-3000}"; SEEDS="${SEEDS:-0 1 2}"; EVAL_N="${EVAL_N:-500}"
mkdir -p runs results
[ -n "${SKIP_AR:-}" ] || python scripts/train_qwen.py --ar-only --eval-n "$EVAL_N" --name baselines
for L in 4 0; do
  for S in $SEEDS; do
    n="${TARGET}_L${L}_s${S}"; mkdir -p "runs/$n"
    python scripts/train_qwen.py --target "$TARGET" --steps "$STEPS" --latent-steps "$L" --seed "$S" \
      --eval-n "$EVAL_N" --eval-every 500 --name "$n" > "runs/$n/train.log" 2>&1 &
    [ "${CONC:-1}" = 1 ] && wait   # one run peaks at ~14GB: only one fits on a 24GB card
  done
  wait
done
touch results/DONE
