"""Compiled (static cache + torch.compile reduce-overhead) latency of the latent+diffusion pipeline, same GPU/prompts/protocol
as bench_latent_diffusion.py (eager) and bench_qwen_ar.py --compiled. One batch size per process (cudagraph trees break when
shapes change inside a process). Also reports how often the compiled output equals the eager (unmerged-LoRA) output.
Usage: python scripts/bench_latent_compiled.py --ckpt runs/<name>/ckpt.pt --batch 1 --out results/latent_compiled_b1.json
"""
import argparse, json, sys

import torch

sys.path.insert(0, "scripts")
from bench_qwen_ar import PROMPTS, timed  # noqa: E402
from train_qwen import ANSWER_SUFFIX, prompt_text  # noqa: E402
from latent_diffusion.qwen_compiled import CompiledLatentDiffusion  # noqa: E402
from latent_diffusion.qwen_latent import QwenLatentDiffusion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--batch", type=int, required=True); ap.add_argument("--latent-mode", default="hidden")
ap.add_argument("--latent-steps", type=int, nargs="+", default=[4, 0]); ap.add_argument("--denoise-steps", type=int, nargs="+", default=[1, 2, 4, 8])
ap.add_argument("--canvas", type=int, default=12); ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--reps", type=int, default=7)
ap.add_argument("--out", required=True)
a = ap.parse_args()

m = QwenLatentDiffusion(steps=4, latent_mode=a.latent_mode).cuda().eval()
m.load_state_dict(torch.load(a.ckpt, map_location="cuda")["model"], strict=False)
enc = m.tok([prompt_text(m.tok, PROMPTS[i % len(PROMPTS)], ANSWER_SUFFIX) for i in range(a.batch)], return_tensors="pt", padding=True).to("cuda")
ids, am = enc.input_ids, enc.attention_mask
cfgs = [(r, d) for r in a.latent_steps for d in a.denoise_steps]
ref = {}
with torch.inference_mode():
    for r, d in cfgs:
        m.steps = r; ref[(r, d)] = m.generate(ids, am, a.canvas, d)   # eager, LoRA still unmerged
    pipe = CompiledLatentDiffusion(m, a.batch, ids.shape[1], canvas=a.canvas, max_latent=max(a.latent_steps))
    rows = []
    for r, d in cfgs:
        for _ in range(a.warmup):
            out = pipe.generate(ids, am, r, d)
        t = sorted(timed(lambda: pipe.generate(ids, am, r, d)) for _ in range(a.reps))
        rows.append(dict(batch=a.batch, latent_steps=r, denoise_steps=d, forward_passes=1 + r + d, e2e_s=t[len(t) // 2],
                         min_s=t[0], match_eager=(out == ref[(r, d)]).float().mean().item(), all_e2e_s=t))
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items() if k != "all_e2e_s"}, flush=True)
json.dump(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, prompt_len=ids.shape[1], compiled=True, rows=rows), open(a.out, "w"), indent=1)
print("wrote", a.out)
