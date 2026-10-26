"""Plan step 2, properly: how much information about the problem is PRESENT in the latents, independent of use?
Linear (ridge) probe from features to the mean token embedding of the question, then retrieval: do a test problem's
features retrieve its own question among all test questions? Features are centered with the train mean.
  latents      : the R latent vectors fed back into the model (concatenated)
  last_hidden  : Qwen's final hidden state at the last prompt token (upper-bound reference: where the latents come from)
  shuffled     : latents with train labels permuted (floor / sanity)
Usage: python scripts/probe_latents.py runs/<name>/ckpt.pt --latent-mode hidden
"""
import argparse, json, sys

import torch
import torch.nn.functional as F
from datasets import load_dataset

sys.path.insert(0, "scripts")
from train_qwen import prompt_text, ANSWER_SUFFIX, trim  # noqa: E402
from latent_diffusion.qwen_latent import QwenLatentDiffusion  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("--latent-mode", default="hidden"); ap.add_argument("--latent-steps", type=int, default=4)
ap.add_argument("--threads", type=int, default=6); ap.add_argument("--n-train", type=int, default=600)
ap.add_argument("--n-test", type=int, default=300); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--out", default="probe_latents.json")
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.manual_seed(0)

m = QwenLatentDiffusion(steps=a.latent_steps, latent_mode=a.latent_mode, dtype=torch.float32).eval()
m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"], strict=False)
ds = load_dataset("openai/gsm8k", "main")


@torch.no_grad()
def extract(questions):
    lat, last, tgt = [], [], []
    for i in range(0, len(questions) - len(questions) % a.bs, a.bs):
        qs = questions[i:i + a.bs]
        enc = m.tok([prompt_text(m.tok, q, ANSWER_SUFFIX) for q in qs], return_tensors="pt", padding=True)
        ids, am = trim(enc.input_ids, enc.attention_mask)
        _, _, _, latents, _ = m.prefix(ids, am)
        lat.append(torch.cat([e[:, 0] for e in latents], dim=-1).float())
        # reference: the hidden state the latent loop starts from (final norm, last prompt token)
        m_steps, m.steps = m.steps, 0
        _, _, _, _, h0 = m.prefix(ids, am); m.steps = m_steps
        last.append(h0.float())
        for q in qs:
            t = m.tok(q, return_tensors="pt", add_special_tokens=False).input_ids
            tgt.append(m._embed(t).mean(1)[0])
        if (i // a.bs) % 10 == 0:
            print("extracted", i + a.bs, flush=True)
    return torch.cat(lat), torch.cat(last), torch.stack(tgt).float()


tq, eq = list(ds["train"]["question"][:a.n_train]), list(ds["test"]["question"][:a.n_test])
Ltr, Htr, Ytr = extract(tq); Lte, Hte, Yte = extract(eq)
n_tr, n_te = len(Ltr), len(Lte)


def center(tr, te):
    mu, sd = tr.mean(0), tr.std(0) + 1e-6
    return (tr - mu) / sd, (te - mu) / sd


def ridge_dual(X, Y, lam):
    K = X @ X.T
    return X.T @ torch.linalg.solve(K + lam * torch.eye(len(X)), Y)


def retrieval(pred, target):
    sims = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).T
    ranks = (sims > sims.diag()[:, None]).sum(1)
    return dict(top1=(ranks == 0).float().mean().item(), top10=(ranks < 10).float().mean().item(),
                median_rank=ranks.float().median().item())


def run(Xtr, Xte, perm=False):
    Xtr, Xte = center(Xtr, Xte); ymu = Ytr.mean(0); Y = Ytr - ymu
    if perm:
        Y = Y[torch.randperm(len(Y))]
    split = int(0.8 * len(Xtr)); best = (None, -1)
    for lam in [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]:          # pick lambda on a held-out slice of train
        W = ridge_dual(Xtr[:split], Y[:split], lam)
        sc = retrieval(Xtr[split:] @ W, Y[split:])["top1"]
        if sc > best[1]:
            best = (lam, sc)
    W = ridge_dual(Xtr, Y, best[0])
    r = retrieval(Xte @ W, Yte - ymu); r["lambda"] = best[0]
    return r


out = {"n_train": n_tr, "n_test": n_te, "chance_top1": 1 / n_te, "chance_top10": 10 / n_te,
       "latents": run(Ltr, Lte), "last_hidden_upper_ref": run(Htr, Hte), "latents_shuffled_labels": run(Ltr, Lte, perm=True)}
# how much of the latent variance is the shared constant? (centered vs raw cosine across problems)
raw = F.normalize(Lte, dim=-1); cen = F.normalize(Lte - Lte.mean(0), dim=-1)
off = lambda x: ((x @ x.T).sum() - len(x)) / (len(x) * (len(x) - 1))
out["latent_cross_problem_cosine"] = dict(raw=off(raw).item(), centered=off(cen).item())
print(json.dumps(out, indent=1)); json.dump(out, open(a.out, "w"), indent=1)
