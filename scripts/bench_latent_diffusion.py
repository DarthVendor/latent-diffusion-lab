"""Latency of the latent-reasoning + diffusion answer pipeline (prompt prefill + R latent steps + D denoising passes over a
fixed answer canvas), timed with CUDA events like bench_qwen_ar.py. Run on the SAME box and session as the AR baselines.
This path is NOT torch.compile-ready (DynamicCache), so compare it with the UNCOMPILED AR numbers; the compiled AR numbers
show what the baseline can do, and forward-pass counts are reported so a compiled estimate can be made. Latency only:
quality is separate (forced-answer Qwen 4.6%, full-reasoning Qwen 62%, this design ~10% on GSM8K).
Usage: python scripts/bench_latent_diffusion.py --ckpt runs/<name>/ckpt.pt --out results/latent_diffusion_latency.json
"""
import argparse, json, platform, sys

import torch
import transformers

sys.path.insert(0, "scripts")
from bench_qwen_ar import PROMPTS, timed  # noqa: E402
from train_qwen import ANSWER_SUFFIX, prompt_text  # noqa: E402
from latent_diffusion.qwen_latent import QwenLatentDiffusion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None); ap.add_argument("--latent-mode", default="hidden")
ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16]); ap.add_argument("--latent-steps", type=int, nargs="+", default=[4, 0])
ap.add_argument("--denoise-steps", type=int, nargs="+", default=[1, 2, 4, 8]); ap.add_argument("--canvas", type=int, default=12)
ap.add_argument("--warmup", type=int, default=2); ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", default="results/latent_diffusion_latency.json")
a = ap.parse_args()

m = QwenLatentDiffusion(steps=4, latent_mode=a.latent_mode).cuda().eval()
if a.ckpt:
    m.load_state_dict(torch.load(a.ckpt, map_location="cuda")["model"], strict=False)
rows = []
with torch.inference_mode():
    for b in a.batches:
        enc = m.tok([prompt_text(m.tok, PROMPTS[i % len(PROMPTS)], ANSWER_SUFFIX) for i in range(b)], return_tensors="pt", padding=True).to("cuda")
        for r in a.latent_steps:
            m.steps = r
            for d in a.denoise_steps:
                for _ in range(a.warmup):
                    m.generate(enc.input_ids, enc.attention_mask, a.canvas, d)
                t = sorted(timed(lambda: m.generate(enc.input_ids, enc.attention_mask, a.canvas, d)) for _ in range(a.reps))
                pre = sorted(timed(lambda: m.prefix(enc.input_ids, enc.attention_mask)) for _ in range(a.reps))
                rows.append(dict(batch=b, latent_steps=r, denoise_steps=d, canvas=a.canvas, prompt_len=enc.input_ids.shape[1],
                                 forward_passes=1 + r + d, e2e_s=t[len(t) // 2], prefix_s=pre[len(pre) // 2],
                                 peak_gb=torch.cuda.max_memory_allocated() / 2**30, all_e2e_s=t))
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "all_e2e_s"}, flush=True)
json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, transformers=transformers.__version__,
               python=platform.python_version(), compiled=False, ckpt=a.ckpt, rows=rows), open(a.out, "w"), indent=1)
print("wrote", a.out)
