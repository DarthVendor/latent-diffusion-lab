"""Plan step 2: what do the latents carry? Test-time prompts (no CoT scaffold). The written solution is predicted from the
latent positions ONLY (prompt hidden), with latents normal / shuffled across problems / zeroed. If shuffling does not raise
the loss, the latents hold nothing problem-specific. Also reports top-1 accuracy on digit tokens (can the numbers be recovered?).
Usage: python scripts/diagnose_latents.py runs/<name>/ckpt.pt --latent-mode hidden [--threads 6] [--n 48]
"""
import argparse, json, re, sys

import torch
from datasets import load_dataset

sys.path.insert(0, "scripts")
from train_qwen import prompt_text, ANSWER_SUFFIX, trim  # noqa: E402
from latent_diffusion.qwen_latent import QwenLatentDiffusion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--latent-mode", default="hidden"); ap.add_argument("--latent-steps", type=int, default=4)
ap.add_argument("--threads", type=int, default=6); ap.add_argument("--n", type=int, default=48); ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--sol-len", type=int, default=160); ap.add_argument("--out", default="diagnose_latents.json")
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)

m = QwenLatentDiffusion(steps=a.latent_steps, latent_mode=a.latent_mode, dtype=torch.float32)
print("unexpected keys:", m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"], strict=False).unexpected_keys[:3], flush=True)
m.eval()
rows = load_dataset("openai/gsm8k", "main")["test"].select(range(a.n))
digit_ids = {i for i in range(m.vocab) if m.tok.decode([i]).strip().isdigit()}
conds = {"normal": None, "shuffled_across_problems": lambda e: e.roll(1, 0), "zeroed": torch.zeros_like}
tot = {c: dict(ce=0.0, n=0, dce=0.0, dn=0, dhit=0) for c in conds}

for i in range(0, a.n - a.n % a.bs, a.bs):
    chunk = rows[i:i + a.bs]
    texts = [prompt_text(m.tok, q, ANSWER_SUFFIX) for q in chunk["question"]]   # test-time prompt: no scaffold
    enc = m.tok(texts, return_tensors="pt", padding=True)
    ids, am = trim(enc.input_ids, enc.attention_mask)
    sols = [m.tok("\n".join(x for x in re.sub(r"<<.*?>>", "", a_.split("####")[0]).strip().split("\n") if x),
                  add_special_tokens=False).input_ids[:a.sol_len] for a_ in chunk["answer"]]
    n = max(map(len, sols))
    sol = torch.tensor([x + [m.pad_id] * (n - len(x)) for x in sols])
    sm = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in sols])
    for name, hook in conds.items():
        m._latent_hook, m._aux_detail = hook, []
        with torch.no_grad():
            cache, pm, start, _, h_last = m.prefix(ids, am)
            m.aux_solution_loss(cache, pm, start, h_last, sol, sm)
        d = m._aux_detail[0]; m._aux_detail, m._latent_hook = None, None
        v = d["valid"].bool(); isd = torch.isin(d["target"], torch.tensor(sorted(digit_ids))) & v
        t = tot[name]
        t["ce"] += d["ce"][v].sum().item(); t["n"] += int(v.sum())
        t["dce"] += d["ce"][isd].sum().item(); t["dn"] += int(isd.sum()); t["dhit"] += int((d["pred"][isd] == d["target"][isd]).sum())
    print("batch", i // a.bs, flush=True)

out = {c: dict(all_token_ce=t["ce"] / t["n"], digit_token_ce=t["dce"] / max(t["dn"], 1),
               digit_token_top1=t["dhit"] / max(t["dn"], 1), n_tokens=t["n"], n_digit_tokens=t["dn"]) for c, t in tot.items()}
print(json.dumps(out, indent=1)); json.dump(out, open(a.out, "w"), indent=1)
