"""Train Qwen3-0.6B + latent steps + masked-diffusion answer canvas on GSM8K final answers, then evaluate.

Canvas = the final numeric answer, padded to --canvas tokens with <|im_end|> (so the model learns where
answers end). Matched comparison: the same prompt ("Answer with only the final number.") goes to plain Qwen
(cached AR, non-thinking). Resumes from runs/<name>/ckpt.pt. Usage: python scripts/train_qwen.py --steps 300
"""
import argparse, json, re, time
from pathlib import Path

import torch
from datasets import load_dataset

from latent_diffusion.qwen_latent import QwenLatentDiffusion

SUFFIX = "\nAnswer with only the final number."


def number(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return m.group(0) if m else None


def encode(tok, rows, canvas, end_id):
    """-> prompt ids/mask (left padded) and targets [N, canvas]; drops answers that do not fit."""
    keep = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", "")) for r in rows]
    keep = [(q, a, tok(a, add_special_tokens=False).input_ids) for q, a in keep]
    keep = [(q, a, t) for q, a, t in keep if len(t) < canvas]
    prompts = [tok.apply_chat_template([{"role": "user", "content": q + SUFFIX}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for q, _, _ in keep]
    enc = tok(prompts, return_tensors="pt", padding=True)
    target = torch.tensor([t + [end_id] * (canvas - len(t)) for _, _, t in keep])
    return enc.input_ids, enc.attention_mask, target, [a for _, a, _ in keep]


@torch.no_grad()
def diffusion_accuracy(m, ids, am, answers, canvas, steps, bs=16):
    m.eval(); ok, t0 = 0, time.time()
    for i in range(0, len(ids), bs):
        out = m.generate(ids[i:i + bs].cuda(), am[i:i + bs].cuda(), canvas, steps)
        for row, gold in zip(out.tolist(), answers[i:i + bs]):
            text = m.tok.decode(row[:row.index(m.end_id)] if m.end_id in row else row)
            ok += number(text) == number(gold)
    torch.cuda.synchronize()
    return ok / len(ids), time.time() - t0


@torch.no_grad()
def ar_accuracy(name, tok, ids, am, answers, bs=16):
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
    ok, t0 = 0, time.time()
    for i in range(0, len(ids), bs):
        out = lm.generate(input_ids=ids[i:i + bs].cuda(), attention_mask=am[i:i + bs].cuda(),
                          max_new_tokens=24, do_sample=False)
        for row, gold in zip(out[:, ids.shape[1]:], answers[i:i + bs]):
            ok += number(tok.decode(row, skip_special_tokens=True)) == number(gold)
    torch.cuda.synchronize()
    return ok / len(ids), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gsm8k_answer")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--latent-steps", type=int, default=4)
    ap.add_argument("--canvas", type=int, default=12)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    root = Path("runs") / a.name; root.mkdir(parents=True, exist_ok=True)

    m = QwenLatentDiffusion(steps=a.latent_steps).cuda()
    lm_tok = m.tok
    m.end_id = lm_tok.convert_tokens_to_ids("<|im_end|>")
    ds = load_dataset("openai/gsm8k", "main")
    tr = encode(lm_tok, ds["train"], a.canvas, m.end_id)
    te = encode(lm_tok, ds["test"].select(range(a.eval_n)), a.canvas, m.end_id)
    print(f"train {len(tr[0])} examples, eval {len(te[0])}, trainable params {sum(p.numel() for p in m.trainable()):,}", flush=True)

    opt = torch.optim.AdamW(m.trainable(), lr=a.lr, weight_decay=0.0)
    step, ckpt = 0, root / "ckpt.pt"
    if ckpt.exists():
        s = torch.load(ckpt, map_location="cuda")
        m.load_state_dict(s["model"], strict=False); opt.load_state_dict(s["opt"]); step = s["step"]
        print("resumed from step", step, flush=True)

    history = []
    while step < a.steps:
        m.train()
        idx = torch.randint(len(tr[0]), (a.bs,))
        loss = m.loss(tr[0][idx].cuda(), tr[1][idx].cuda(), tr[2][idx].cuda())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.trainable(), 1.0); opt.step(); step += 1
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            acc, secs = diffusion_accuracy(m, te[0], te[1], te[3], a.canvas, steps=8)
            history.append(dict(step=step, loss=loss.item(), diffusion_exact=acc, eval_s=secs))
            print(history[-1], flush=True)
            torch.save({"model": m.extra_state(), "opt": opt.state_dict(), "step": step}, ckpt)

    final = {"history": history, "args": vars(a)}
    for s in (2, 4, 8):
        acc, secs = diffusion_accuracy(m, te[0], te[1], te[3], a.canvas, steps=s)
        final[f"diffusion_steps_{s}"] = dict(exact=acc, seconds=secs)
    acc, secs = ar_accuracy("Qwen/Qwen3-0.6B", lm_tok, te[0], te[1], te[3])
    final["qwen_ar_direct"] = dict(exact=acc, seconds=secs)
    final["note"] = "seconds are uncompiled wall-clock incl. python overhead: NOT comparable to bench_qwen_ar numbers"
    (root / "results.json").write_text(json.dumps(final, indent=1))
    print(json.dumps(final, indent=1))
