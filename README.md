# Latent Diffusion Lab

A **research prototype** testing whether SLT-style continuous latent reasoning can
condition a **bidirectional masked-token diffusion** decoder. Independent implementation,
not the original Soft Latent Thinking authors' repository and not a reproduction of their results.

## What works today

- Trainable continuous latent projector: hidden state -> K basis logits ->
  Gumbel-softmax mixture -> recurrent hidden-state update.
- Fixed-size multi-query cross-attention compression of the reasoning trajectory.
- Bidirectional transformer denoiser, masked reconstruction loss, cosine-free
  confidence-based progressive unmasking of answer positions.
- Controlled **without-latent vs with-latent** architectural ablation, identical evaluation
  split, deterministic seeding, periodic metrics and checkpoints.
- Optional `QwenReasoner` backbone adapter. Requires Hugging Face weights and extra
  dependencies and is **not** wired into the two-digit training CLI by default.
- CPU smoke test; pytest tests; original standalone toy preserved in `examples/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest -q
```

## Run an experiment

```bash
ldlab train --config configs/smoke.yaml
ldlab train --config configs/addition.yaml --iters 500
# artifacts: runs/addition/results.json, baseline.pt, latent_diffusion.pt
```

This first task uses **80/20 held-out digit pairs** for zero-padded two-digit addition.
It is an interface/infrastructure check, **not proof of improved reasoning**.
Models can memorize training examples; compare held-out accuracy, different random
splits, and multiple seeds before drawing conclusions.

## Architecture

```text
Prompt IDs
   -> causal prompt causal transformer
   -> initial h
   -> [linear K + Gumbel softmax + learned basis + causal transformerCell] x R (optional)
   -> latent trajectory [B,R+1,D]
   -> learned attention pooling [B,M,D]
   -> masked target IDs + diffusion timestep embedding
   -> bidirectional transformer over [memory, noisy answer]
   -> per-position token logits
   -> confidence-based iterative unmasking
```

**Technical limits:** The transformer-based latent steps are only *inspired* by SLT; they do
not append soft embeddings through all original Qwen transformer layers. Likewise,
the decoder uses a MaskGIT-style unmasking schedule and a simplified masked
cross-entropy objective, not an exact reproduction of LLaDA. The two backbone
choices are not parameter matched: the no-latent ablation leaves projector and
transition weights unused. This repo does not claim a speedup or SOTA accuracy.

## Pretrained Qwen extension

```bash
pip install -e '.[qwen]'
```

`QwenReasoner` in `src/latent_diffusion/models.py` can load a Hugging Face
Qwen model via `AutoModel.from_pretrained(model_name)`, extract its final prompt
state, perform trainable SLT-like recurrence, and return projected reasoning
states. **It is an experimental adapter, not yet a pretrained end-to-end
Qwen+diffusion training recipe.** In particular, its recurrent latent states
are not fed through every transformer block; converting the decoder into a
pretrained Qwen diffusion model, tokenization/data collation, LoRA, and
training on mathematical corpora remain future work.

## Repo layout

```text
src/latent_diffusion/  # model, diffusion sampler, synthetic dataset, training, CLI
configs/               # small CPU smoke and longer baseline experiment
scripts/               # shell examples
examples/original_toy.py
runs/                  # generated only, gitignored
tests/                 # unit/integration tests
```

## Suggested experiments

1. Run ≥5 seeds, report mean±std held-out exact accuracy and runtime.
2. Ablate memory slots (1,4,16) and denoising steps (1,2,4,8).
3. Replace synthetic digits with a tokenizer and a pretrained conditional
   masked decoder; ensure attention masks prevent target leakage.
4. Compare decoder from prompt alone vs last reasoner state vs full trajectory.
5. Profile full-pass FLOPs/latency separately from number of iterative passes.

## References

- *A Model with No Head and Many Thoughts: Soft Latent Thinking*, arXiv:2608.31069.
- *Large Language Diffusion Models*, arXiv:2502.09992 (LLaDA).

This code is released under MIT. Pretrained Qwen weights have their own licenses.

## Verified local run

The four pytest tests passed locally. A 200-step CPU addition run (single seed, one split) yielded:

| Model | Train exact | Held-out exact |
|---|---:|---:|
| No latent | 82.5% | 45.0% |
| SLT-inspired + diffusion | 97.5% | 35.0% |

These unstable, small-data results are **not evidence of a benefit** from latent reasoning. See `results/validation200.json`; checkpoints from this specific run are excluded from the distributed ZIP to keep it small.

## October 2026 recurrent-transformer experiment

The toy reasoner now runs a **causal transformer** on the growing prompt/soft-token sequence, feeding each learned continuous latent embedding back through the same transformer on every step. This is no longer a GRU state-update loop. The optional pretrained Qwen adapter is still experimental and is not used by the main training CLI.

Training fixes: 75% forced fully masked outputs, mask timestep recomputed from actual corruption, deterministic latent training/evaluation by default, lower learning rate (3e-4), and checkpoints including AdamW state. See `results/recurrent_transformer_250_summary.json` for the single-seed CPU run: at step 250, baseline validation accuracy was 5% and recurrent-transformer validation accuracy was 60% on a 20-example held-out set. **Do not interpret this as validated generalization**; run repeated seeds and larger arithmetic datasets next.
