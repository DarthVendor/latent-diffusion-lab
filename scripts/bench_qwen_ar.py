"""Baseline: Qwen3-0.6B with cached autoregressive decoding, timed with CUDA events.

Fixed-length timing: min_new_tokens == max_new_tokens, so EOS never shortens a run.
This measures latency/memory only; answer quality is NOT measured here.
Usage: python scripts/bench_qwen_ar.py --smoke | [--compiled] --out results/ar_baseline.json
Report both eager and --compiled; the compiled run is the "optimized cached AR" baseline the plan requires.
Warmup absorbs compile time (one compile per batch/length shape), so it is never in the timings.
"""
import argparse, json, platform, time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?",
    "A train travels 60 km in 45 minutes. At the same speed, how far does it travel in 2 hours?",
    "If 3x + 7 = 25, what is x? Explain briefly.",
    "Tom has 5 boxes with 12 apples each and gives away 17 apples. How many apples are left?",
]


def timed(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / 1000  # seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (default: non-thinking)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--compiled", action="store_true",
                    help="static KV cache + torch.compile(reduce-overhead): removes HF eager launch overhead")
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the GPU path works")
    ap.add_argument("--out", default="results/ar_baseline.json")
    a = ap.parse_args()
    if a.smoke:
        a.batches, a.lengths, a.warmup, a.reps = [1], [32], 1, 1
    assert torch.cuda.is_available(), "CUDA GPU required"

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if a.compiled:
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    torch.cuda.synchronize()
    load_s = time.time() - t0

    def prompt_batch(b):
        texts = [tok.apply_chat_template([{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=a.thinking) for i in range(b)]
        return tok(texts, return_tensors="pt", padding=True).to("cuda")

    rows = []
    for b in a.batches:
        inp = prompt_batch(b)
        for n in a.lengths:
            gen = lambda: model.generate(**inp, max_new_tokens=n, min_new_tokens=n, do_sample=False, use_cache=True,
                                    **({"cache_implementation": "static"} if a.compiled else {}))
            with torch.inference_mode():
                for _ in range(a.warmup):
                    gen()
                pre = [timed(lambda: model(**inp, use_cache=True)) for _ in range(a.reps)]
                torch.cuda.reset_peak_memory_stats()
                tot = [timed(gen) for _ in range(a.reps)]
            med = sorted(tot)[len(tot) // 2]
            rows.append(dict(batch=b, new_tokens=n, prompt_len=inp.input_ids.shape[1], e2e_s=med,
                             prefill_s=sorted(pre)[len(pre) // 2], tok_per_s=b * n / med,
                             peak_gb=torch.cuda.max_memory_allocated() / 2**30, all_e2e_s=tot))
            print(rows[-1], flush=True)

    out = dict(model=a.model, thinking=a.thinking, compiled=a.compiled, dtype="bfloat16", attn="sdpa", load_s=load_s,
               gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, torch=torch.__version__,
               transformers=transformers.__version__, python=platform.python_version(), rows=rows)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
