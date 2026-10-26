"""Train Qwen3-0.6B + latent steps + masked-diffusion answer canvas on GSM8K, then evaluate.

--target answer    canvas = final number only (default; compare with Qwen's "forced" direct answer)
--target solution  canvas = worked solution ending "#### n" (compare with Qwen's full reasoning)
--latent-steps 0   ablation: same model and training, no latent reasoning steps
Canvas is padded with <|im_end|> so the model learns where answers end. Resumes from runs/<name>/ckpt.pt.
Examples: python scripts/train_qwen.py --steps 3000 --name a_s0
          python scripts/train_qwen.py --ar-only --eval-n 500        # Qwen baselines, once
"""
import argparse, json, re, time
from pathlib import Path

import torch
from datasets import load_dataset

from latent_diffusion.qwen_latent import QwenLatentDiffusion

ANSWER_SUFFIX = "\nAnswer with only the final number."
NUM = r"-?\d+(?:\.\d+)?"


def number(s, last=False):
    s = s.replace(",", "")
    if "####" in s:
        s = s.split("####")[-1]
    ms = re.findall(NUM, s)
    return (ms[-1] if last else ms[0]) if ms else None


def gold_text(row, target):
    sol, ans = row["answer"].split("####")
    ans = ans.strip().replace(",", "")
    return ans if target == "answer" else re.sub(r"<<.*?>>", "", sol).strip() + f"\n#### {ans}"


def prompt_text(tok, q, suffix, prefill=""):
    return tok.apply_chat_template([{"role": "user", "content": q + suffix}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False) + prefill


def encode(tok, rows, target, canvas, end_id):
    """-> prompt ids/mask (left padded), targets [N, canvas], gold answers; drops examples that do not fit."""
    suffix = ANSWER_SUFFIX if target == "answer" else ""
    items = [(r["question"], r["answer"].split("####")[-1].strip().replace(",", ""),
              tok(gold_text(r, target), add_special_tokens=False).input_ids) for r in rows]
    items = [i for i in items if len(i[2]) < canvas]
    enc = tok([prompt_text(tok, q, suffix) for q, _, _ in items], return_tensors="pt", padding=True)
    target_ids = torch.tensor([t + [end_id] * (canvas - len(t)) for _, _, t in items])
    return enc.input_ids, enc.attention_mask, target_ids, [a for _, a, _ in items], [q for q, _, _ in items]


def trim(ids, am):
    """Drop left-padding columns that are padding for every row in this batch (prompts were padded to the dataset max)."""
    first = am.any(0).nonzero()[0, 0]
    return ids[:, first:], am[:, first:]


@torch.no_grad()
def diffusion_accuracy(m, data, canvas, steps, target, bs=16):
    m.eval(); ok, t0 = 0, time.time()
    ids, am, _, answers, _ = data
    for i in range(0, len(ids), bs):
        out = m.generate(*(x.cuda() for x in trim(ids[i:i + bs], am[i:i + bs])), canvas, steps)
        for row, gold in zip(out.tolist(), answers[i:i + bs]):
            text = m.tok.decode(row[:row.index(m.end_id)] if m.end_id in row else row)
            ok += number(text, last=target == "solution") == number(gold)
    torch.cuda.synchronize()
    return ok / len(ids), time.time() - t0


@torch.no_grad()
def ar_accuracy(name, tok, questions, answers, mode, bs=16):
    """Plain Qwen, cached AR, non-thinking. forced: answer prefilled ("The final answer is"), 16 new tokens.
    reasoning: free chain of thought, 512 new tokens, last number scored."""
    from transformers import AutoModelForCausalLM
    lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
    suffix, prefill, new = (ANSWER_SUFFIX, "The final answer is ", 16) if mode == "forced" else (
        "\nThink step by step, then finish with 'The final answer is N'.", "", 512)
    ok, t0 = 0, time.time()
    for i in range(0, len(questions), bs):
        enc = tok([prompt_text(tok, q, suffix, prefill) for q in questions[i:i + bs]],
                  return_tensors="pt", padding=True).to("cuda")
        out = lm.generate(**enc, max_new_tokens=new, do_sample=False)
        for row, gold in zip(out[:, enc.input_ids.shape[1]:], answers[i:i + bs]):
            text = tok.decode(row, skip_special_tokens=True)
            ok += number(text, last=mode == "reasoning") == number(gold)
    torch.cuda.synchronize()
    return dict(exact=ok / len(questions), seconds=time.time() - t0, n=len(questions))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="gsm8k_answer")
    ap.add_argument("--target", choices=["answer", "solution"], default="answer")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--latent-steps", type=int, default=4)
    ap.add_argument("--canvas", type=int, default=None, help="default 12 (answer) / 256 (solution)")
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ar-only", action="store_true", help="only run the plain-Qwen baselines")
    a = ap.parse_args()
    a.canvas = a.canvas or (12 if a.target == "answer" else 256)
    torch.manual_seed(a.seed)
    root = Path("runs") / a.name; root.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")

    if a.ar_only:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
        rows = ds["test"].select(range(a.eval_n))
        qs, golds = list(rows["question"]), [x.split("####")[-1].strip().replace(",", "") for x in rows["answer"]]
        out = {mode: ar_accuracy("Qwen/Qwen3-0.6B", tok, qs, golds, mode) for mode in ("forced", "reasoning")}
        (root / "ar_baselines.json").write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))
        raise SystemExit

    m = QwenLatentDiffusion(steps=a.latent_steps).cuda()
    m.end_id = m.tok.convert_tokens_to_ids("<|im_end|>")
    tr = encode(m.tok, ds["train"], a.target, a.canvas, m.end_id)
    te = encode(m.tok, ds["test"].select(range(a.eval_n)), a.target, a.canvas, m.end_id)
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
        loss = m.loss(*(x.cuda() for x in trim(tr[0][idx], tr[1][idx])), tr[2][idx].cuda())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.trainable(), 1.0); opt.step(); step += 1
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            acc, secs = diffusion_accuracy(m, te, a.canvas, 8, a.target)
            history.append(dict(step=step, loss=loss.item(), diffusion_exact=acc, eval_s=secs))
            print(history[-1], flush=True)
            torch.save({"model": m.extra_state(), "opt": opt.state_dict(), "step": step}, ckpt)

    final = {"history": history, "args": vars(a), "note": "seconds are uncompiled wall-clock: NOT comparable to bench_qwen_ar"}
    for s in (2, 4, 8):
        acc, secs = diffusion_accuracy(m, te, a.canvas, s, a.target)
        final[f"diffusion_steps_{s}"] = dict(exact=acc, seconds=secs, n=len(te[0]))
    (root / "results.json").write_text(json.dumps(final, indent=1)); print(json.dumps(final, indent=1))
