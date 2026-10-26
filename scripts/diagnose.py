"""Why does accuracy plateau? Diagnose a trained checkpoint (CPU-friendly, small batches).
 A. gradient flow: per-group grad norm, and signal fraction ||mean grad|| / mean ||grad|| over batches
 B. do the latents matter: masked loss / exact match with latents normal vs zeroed vs shuffled across examples vs removed
 C. latent statistics: basis-weight entropy and cross-example similarity (collapse check)
 D. per-token loss on a fully masked canvas: answer digits vs <|im_end|> fill
Usage: python scripts/diagnose.py runs/<name>/ckpt.pt --threads 6
"""
import argparse, json, math, sys

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "scripts")
from train_qwen import encode, trim, number  # noqa: E402
from latent_diffusion.diffusion import corrupt  # noqa: E402
from latent_diffusion.qwen_latent import QwenLatentDiffusion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--threads", type=int, default=6)
ap.add_argument("--bs", type=int, default=4); ap.add_argument("--batches", type=int, default=12)
ap.add_argument("--eval-n", type=int, default=48); ap.add_argument("--latent-steps", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)

m = QwenLatentDiffusion(steps=a.latent_steps, dtype=torch.float32)
print("load:", m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"], strict=False).unexpected_keys[:3], flush=True)
m.end_id = m.tok.convert_tokens_to_ids("<|im_end|>")
ds = load_dataset("openai/gsm8k", "main")
tr = encode(m.tok, ds["train"].select(range(600)), "answer", 12, m.end_id)
te = encode(m.tok, ds["test"].select(range(a.eval_n)), "answer", 12, m.end_id)
out = {}


def batch(data, i):
    sl = slice(i * a.bs, (i + 1) * a.bs)
    return (*trim(data[0][sl], data[1][sl]), data[2][sl])

# ---- A. gradient flow -------------------------------------------------------------------------------
def group(n):
    return ("lora" if "lora_" in n else "project/norm" if n.startswith(("project", "norm")) else
            "basis" if n == "basis" else "mask_vec" if n == "mask_vec" else None)

m.train(); grads = {}
for i in range(a.batches):
    m.zero_grad(set_to_none=True)
    ids, am, tgt = batch(tr, i)
    m.loss(ids, am, tgt).backward()
    for n, p in m.named_parameters():
        g = group(n)
        if g and p.grad is not None:
            grads.setdefault(g, [[] for _ in range(a.batches)])[i].append(p.grad.flatten().clone())
    print("grad batch", i, flush=True)
out["A_gradients"] = {}
for g, per in grads.items():
    vecs = torch.stack([torch.cat(v) for v in per])  # [batches, params]
    norms = vecs.norm(dim=1)
    cos = F.cosine_similarity(vecs[:, None], vecs[None], dim=-1)
    off = cos[~torch.eye(len(vecs), dtype=torch.bool)].mean().item()
    out["A_gradients"][g] = dict(mean_grad_norm=norms.mean().item(), signal_fraction=(vecs.mean(0).norm() / norms.mean()).item(),
                                 mean_cos_between_batches=off, n_params=vecs.shape[1])
print(json.dumps(out["A_gradients"], indent=1), flush=True)

# ---- B/C/D need no grad -------------------------------------------------------------------------------
m.eval()


@torch.no_grad()
def masked_loss(data, hook=None, steps=None, nb=6, p_full=1.0):
    m._latent_hook, old = hook, m.steps
    if steps is not None:
        m.steps = steps
    tot, cnt = 0.0, 0
    for i in range(nb):
        ids, am, tgt = batch(data, i)
        cache, pm, start, *_ = m.prefix(ids, am)
        torch.manual_seed(100 + i)
        noisy, masks, _ = corrupt(tgt, m.mask_id, m.pad_id, full_mask_prob=p_full)
        h = m.canvas_hidden(cache, pm, start, noisy)
        lg = m.lm.get_output_embeddings()(h[masks])[:, :m.vocab]
        tot += F.cross_entropy(lg, tgt[masks], reduction="sum").item(); cnt += int(masks.sum())
    m._latent_hook, m.steps = None, old
    return tot / cnt


@torch.no_grad()
def exact(data, hook=None, steps=None):
    m._latent_hook, old = hook, m.steps
    if steps is not None:
        m.steps = steps
    ok = 0
    for i in range(0, len(data[0]) // a.bs * a.bs, a.bs):
        ids, am = trim(data[0][i:i + a.bs], data[1][i:i + a.bs])
        for row, gold in zip(m.generate(ids, am, 12, 8).tolist(), data[3][i:i + a.bs]):
            ok += number(m.tok.decode(row[:row.index(m.end_id)] if m.end_id in row else row)) == number(gold)
    m._latent_hook, m.steps = None, old
    return ok / (len(data[0]) // a.bs * a.bs)


conds = {"normal": (None, None), "latents_zeroed": (torch.zeros_like, None),
         "latents_shuffled_across_examples": (lambda e: e.roll(1, 0), None), "no_latent_steps": (None, 0)}
out["B_latent_usefulness"] = {}
for name, (hook, steps) in conds.items():
    out["B_latent_usefulness"][name] = dict(test_masked_loss_full_mask=masked_loss(te, hook, steps), test_exact=exact(te, hook, steps))
    print(name, out["B_latent_usefulness"][name], flush=True)

m._trace = []
with torch.no_grad():
    ids, am, tgt = batch(te, 0)
    m.prefix(ids, am)
tr_ = m._trace; m._trace = None
out["C_latents"] = []
for r, (w, e) in enumerate(tr_):
    ent = -(w * (w + 1e-9).log()).sum(-1).mean().item()
    en = F.normalize(e[:, 0].float(), dim=-1)
    sim = (en @ en.T)[~torch.eye(len(en), dtype=torch.bool)].mean().item()
    out["C_latents"].append(dict(step=r, weight_entropy_nats=ent, max_entropy=math.log(w.shape[-1]),
                                 top_weight=w.max(-1)[0].mean().item(), cross_example_cosine=sim, norm=e.float().norm(dim=-1).mean().item()))
print(json.dumps(out["C_latents"], indent=1), flush=True)

with torch.no_grad():
    dig, fill, dig_ok, dig_n = 0.0, 0.0, 0, 0
    nd = nf = 0
    for i in range(6):
        ids, am, tgt = batch(te, i)
        cache, pm, start, *_ = m.prefix(ids, am)
        h = m.canvas_hidden(cache, pm, start, torch.full_like(tgt, m.mask_id))
        lg = m.lm.get_output_embeddings()(h)[..., :m.vocab]
        ce = F.cross_entropy(lg.transpose(1, 2), tgt, reduction="none")
        isf = tgt.eq(m.end_id)
        dig += ce[~isf].sum().item(); nd += int((~isf).sum()); fill += ce[isf].sum().item(); nf += int(isf.sum())
        dig_ok += int((lg.argmax(-1).eq(tgt) & ~isf).sum())
out["D_per_token_full_mask"] = dict(answer_token_loss=dig / nd, fill_token_loss=fill / nf, answer_token_top1=dig_ok / nd, n_answer_tokens=nd, n_fill_tokens=nf)
print(json.dumps(out["D_per_token_full_mask"], indent=1))
json.dump(out, open("diagnose.json", "w"), indent=1)
