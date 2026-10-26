# Qwen3-0.6B latent reasoning → masked diffusion: experiment plan

## Narrow project scope
**Primary target:** `Qwen/Qwen3-0.6B` (post-trained) initialized from the same pretrained checkpoint for every matched comparison. Develop **one** experimental model: Qwen3-0.6B with SLT-inspired continuous reasoning and masked diffusion final-answer generation. Retain prior toy code as a diagnostic, not evidence of target-model performance.

**Comparators (not additional research models to invent):**
1. Standard **Qwen3-0.6B** with optimized autoregressive decoding (causal attention, RoPE, KV cache, matched hardware/precision/sequence settings).
2. **Soft Latent Thinking (SLT)** as the directly related method, if compatible weights/code and matched 0.6B reproduction are available. Otherwise compare to *reported* SLT results qualitatively only; do not compare across models as if controlled.
3. Optional implementation reference: DiffusionLLM Qwen3 AR→masked-diffusion conversion, which **requires diffusion fine-tuning**. It is a conversion baseline/engineering reference, not equivalent to our full latent-reasoning proposal.

## Architecture
- Load Qwen3-0.6B weights, tokenizer, native RoPE/GQA and LM head. Keep the standard Qwen inference path independently runnable.
- SLT-style learnable projector produces continuous embeddings; append each through `inputs_embeds` with proper positional IDs and a correct recurrent transformer KV cache. Verify cached vs uncached prefix results.
- Define a deterministic transition/handoff to a fixed-length or blockwise masked output canvas; allow output region bidirectional attention, but **prevent reasoning memory from attending to noisy output tokens**.
- Preserve prefix KV cache across diffusion steps and recompute only active denoising positions where mathematically correct. Confirm equivalence to an uncached full-sequence reference.
- Reuse pretrained Qwen weights for diffusion conversion; add a mask token and fine-tune on genuine masked-token objectives. A causal-mask toggle alone is not a functioning diffusion LM.
- Start with LoRA and BF16 on one NVIDIA GPU; carefully version attention-mask logic, cache formats, RoPE positions, tokenizer expansion and Qwen Transformers version. Do not imply the old GRU/toy training pipeline is equivalent.

## Evaluation: cost / speed *at acceptable quality*
- Same Qwen3-0.6B checkpoint, tokenization, prompts, hardware, batch sizes (1,4,16), precision and GPU benchmark harness.
- Measure **end-to-end time to answer** (prefill + latent thinking + decoding), decoder-only latency, peak CUDA memory, throughput, GPU-seconds per successful answer, and answer quality. Include loading/warmup and CUDA-event synchronization protocol. Report GPU model, driver/runtime and software versions.
- Vary generated answer lengths (32,64,128,256,512), diffusion steps (2,4,8,16,32), and latent reasoning steps (0,4,8,16). Include cost/quality Pareto frontier; no accuracy claims from random-weight timing.
- For fairness, compare normal Qwen including its textual thinking budget where relevant and also its non-thinking mode. Report how reasoning-token budget maps to latent steps; do not claim equivalence of one latent step and one text token.
- Use small held-out language/reasoning datasets, not only 2-digit addition. Report exact-match when suitable plus task-specific quality metrics; validate both answer quality and length completion.
- Distinguish measured numbers from projected costs or previously obtained toy CPU results.

## Success criteria
Demonstrate a reproducible **end-to-end GPU** latency/cost gain at comparable downstream answer quality against optimized baseline Qwen3-0.6B. Run >=3 random training seeds or checkpoints when feasible. If diffusion does not match quality, report the quality–latency frontier rather than claiming a speedup.

## Agent review checklist
- [ ] Verify repo code is truly Qwen3-backed, rather than the existing synthetic GRU/toy backbone.
- [ ] Audit RoPE position offsets, GQA heads, prefix-cache reuse, attention masking/leakage.
- [ ] Audit correct masked-diffusion objective, remasking schedule and termination/length handling.
- [ ] Check that baseline Qwen uses optimized cached AR and identical inference conditions.
- [ ] Verify project can launch on Vast.ai with reproducible pinned dependencies, checkpoint resume, GPU test and budget limits.
- [ ] Check related work: SLT (arXiv:2608.31069), DiffusionLLM (lamm-mit/DiffusionLLM), Block Diffusion, CCDD, LaDiR.
- [ ] Flag unsupported claims from prior toy CPU benchmarks; document what is and is not demonstrated.
- [ ] Produce a concise code-review report with blockers, prioritized tasks, and smoke-test commands.

## References
- Qwen3-0.6B: https://huggingface.co/Qwen/Qwen3-0.6B
- SLT: https://arxiv.org/abs/2608.31069
- DiffusionLLM conversion: https://github.com/lamm-mit/DiffusionLLM

**Status:** Research scope/design only. No claim that pretrained Qwen latent + diffusion has been implemented or benchmarked yet.
